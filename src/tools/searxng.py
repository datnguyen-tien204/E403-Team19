import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Literal, Optional
from urllib.parse import urlparse

import requests
from readability import Document
from markdownify import markdownify as md

ExtractMode = Literal["markdown", "text"]

DEFAULT_FETCH_MAX_CHARS = 200_000
DEFAULT_FETCH_MAX_RESPONSE_BYTES = 2_000_000
DEFAULT_FETCH_MAX_REDIRECTS = 3
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_CACHE_TTL_SECONDS = 300
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

_FETCH_CACHE: dict[str, tuple[float, dict]] = {}


@dataclass
class WebFetchConfig:
    max_chars: int = DEFAULT_FETCH_MAX_CHARS
    max_response_bytes: int = DEFAULT_FETCH_MAX_RESPONSE_BYTES
    max_redirects: int = DEFAULT_FETCH_MAX_REDIRECTS
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS
    user_agent: str = DEFAULT_USER_AGENT
    readability_enabled: bool = True


def _cache_key(url: str, extract_mode: str, max_chars: int) -> str:
    raw = f"fetch:{url}:{extract_mode}:{max_chars}"
    return hashlib.md5(raw.encode()).hexdigest()


def _read_cache(key: str) -> Optional[dict]:
    entry = _FETCH_CACHE.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.time() > expires_at:
        del _FETCH_CACHE[key]
        return None
    return value


def _write_cache(key: str, value: dict, ttl_seconds: int):
    _FETCH_CACHE[key] = (time.time() + ttl_seconds, value)


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _html_to_markdown(html: str, url: str = "") -> tuple[str, Optional[str]]:
    try:
        doc = Document(html)
        title = doc.title()
        content_html = doc.summary()
        markdown = md(content_html, heading_style="ATX")
        return markdown.strip(), title
    except Exception:
        return "", None


def _markdown_to_text(markdown: str) -> str:
    import re
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", markdown)
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\*{1,2}([^\*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_content(html: str, url: str, extract_mode: ExtractMode) -> tuple[str, Optional[str]]:
    markdown, title = _html_to_markdown(html, url)
    if not markdown:
        return "", title
    if extract_mode == "text":
        return _markdown_to_text(markdown), title
    return markdown, title


def fetch_url(
    url: str,
    extract_mode: ExtractMode = "markdown",
    max_chars: int = DEFAULT_FETCH_MAX_CHARS,
    config: Optional[WebFetchConfig] = None,
) -> dict:
    cfg = config or WebFetchConfig()

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Invalid URL: must be http or https")

    cache_key = _cache_key(url, extract_mode, max_chars)
    cached = _read_cache(cache_key)
    if cached:
        return {**cached, "cached": True}

    start = time.time()

    session = requests.Session()
    session.max_redirects = cfg.max_redirects

    resp = session.get(
        url,
        timeout=cfg.timeout_seconds,
        headers={
            "Accept": "text/markdown, text/html;q=0.9, */*;q=0.1",
            "User-Agent": cfg.user_agent,
            "Accept-Language": "en-US,en;q=0.9",
        },
        stream=True,
    )

    final_url = resp.url

    # Read body với giới hạn bytes
    chunks = []
    total = 0
    truncated_response = False
    for chunk in resp.iter_content(chunk_size=8192):
        chunks.append(chunk)
        total += len(chunk)
        if total >= cfg.max_response_bytes:
            truncated_response = True
            break
    body_bytes = b"".join(chunks)
    body = body_bytes.decode("utf-8", errors="replace")

    content_type = resp.headers.get("content-type", "application/octet-stream")
    title = None
    extractor = "raw"
    text = body

    if not resp.ok:
        raise RuntimeError(f"Web fetch failed ({resp.status_code}): {resp.reason}")

    if "text/markdown" in content_type:
        extractor = "cf-markdown"
        if extract_mode == "text":
            text = _markdown_to_text(body)

    elif "text/html" in content_type:
        if cfg.readability_enabled:
            extracted, title = _extract_content(body, final_url, extract_mode)
            if extracted:
                text = extracted
                extractor = "readability"
            else:
                raise RuntimeError("Web fetch extraction failed: Readability returned no content.")
        else:
            raise RuntimeError("Web fetch extraction failed: Readability disabled.")

    elif "application/json" in content_type:
        try:
            text = json.dumps(json.loads(body), indent=2, ensure_ascii=False)
            extractor = "json"
        except Exception:
            text = body
            extractor = "raw"

    text_truncated, was_truncated = _truncate_text(text, max_chars)

    warning = None
    if truncated_response:
        warning = f"Response body truncated after {cfg.max_response_bytes} bytes."

    payload = {
        "url": url,
        "final_url": final_url,
        "status": resp.status_code,
        "content_type": content_type.split(";")[0].strip(),
        "title": title,
        "extract_mode": extract_mode,
        "extractor": extractor,
        "truncated": was_truncated,
        "raw_length": len(text),
        "length": len(text_truncated),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "took_ms": int((time.time() - start) * 1000),
        "text": text_truncated,
        "warning": warning,
    }

    _write_cache(cache_key, payload, cfg.cache_ttl_seconds)
    return payload


def search_and_fetch(
    searxng_url: str,
    query: str,
    top_k: int = 3,
    extract_mode: ExtractMode = "markdown",
    config: Optional[WebFetchConfig] = None,
    max_workers: int = 5,
) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    resp = requests.get(
        f"{searxng_url}/search",
        params={"q": query, "format": "json"},
        timeout=10,
    )
    results = resp.json()["results"][:top_k]

    def fetch_one(r: dict) -> dict:
        item = {
            "title": r["title"],
            "url": r["url"],
            "snippet": r["content"],
            "full_content": None,
            "error": None,
        }
        try:
            item["full_content"] = fetch_url(r["url"], extract_mode=extract_mode, config=config)
        except Exception as e:
            item["error"] = str(e)
        return item

    output = [None] * len(results)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, r): i for i, r in enumerate(results)}
        for future in as_completed(futures):
            idx = futures[future]
            output[idx] = future.result()

    return output


if __name__ == "__main__":
    results = search_and_fetch(
        searxng_url="http://localhost:8888",
        query="Tập Cận Bình",
        top_k=3,
        extract_mode="markdown",
    )
    for i, r in enumerate(results):
        print(f"\n{'='*60}")
        print(f"[{i+1}] {r['title']}")
        print(f"URL: {r['url']}")
        if r["full_content"]:
            fc = r["full_content"]
            print(f"Extractor: {fc['extractor']} | Took: {fc['took_ms']}ms")
            print(f"Content ({fc['length']} chars):\n{fc['text']}")
        else:
            print(f"Error: {r['error']}")