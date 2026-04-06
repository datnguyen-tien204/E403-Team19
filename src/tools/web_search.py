"""
tools/web_search.py
Web search tool – SearxNG primary, DuckDuckGo fallback.

Improvements:
  ✅ Retry với backoff khi SearxNG timeout
  ✅ Dedup URL trước khi fetch (tránh tốn thời gian fetch trùng)
  ✅ Timeout riêng cho từng URL fetch, không block cả batch
  ✅ Snippet fallback nếu fetch nội dung thất bại (thay vì bỏ trống)
  ✅ Ghi rõ nguồn lỗi để agent biết dùng kết quả nào
  ✅ Lọc kết quả rác (nội dung quá ngắn < 200 ký tự)
  ✅ Trả về metadata (domain, độ dài) giúp LLM đánh giá độ tin cậy
"""
from __future__ import annotations

import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from typing import Optional
from urllib.parse import urlparse

from langchain_core.tools import tool
from duckduckgo_search import DDGS

from src.config import WEB_SEARCH_TOP_K, SEARXNG_URL
from src.tools.searxng import search_and_fetch, fetch_url, WebFetchConfig

# ── Fetch config: timeout chặt hơn để không block quá lâu ────────────────────
_FETCH_CFG = WebFetchConfig(
    timeout_seconds=15,          # mỗi URL tối đa 15s (thay vì 30s mặc định)
    max_response_bytes=500_000,  # 500KB là đủ cho hầu hết bài viết
    max_chars=8_000,             # giới hạn text trả về LLM (~2k tokens)
    cache_ttl_seconds=300,
)

_SEARXNG_URL = SEARXNG_URL        # đọc từ config (env: SEARXNG_URL)
_MIN_CONTENT_LEN = 200   # bỏ kết quả nội dung quá ngắn (quảng cáo, 404…)


def _check_searxng_health() -> tuple[bool, str]:
    """Ping SearXNG /healthz để kiểm tra trước khi search."""
    try:
        r = requests.get(f"{_SEARXNG_URL}/healthz", timeout=3)
        if r.ok:
            return True, "ok"
        return False, f"HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"Không kết nối được tới {_SEARXNG_URL} – kiểm tra SearXNG có đang chạy không, và SEARXNG_URL trong .env"
    except requests.exceptions.Timeout:
        return False, f"Timeout khi ping {_SEARXNG_URL}"
    except Exception as e:
        return False, str(e)


def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url


def _fetch_one_safe(url: str, title: str, snippet: str) -> dict:
    """Fetch URL, trả về dict chuẩn, không raise exception ra ngoài."""
    result = {
        "title": title,
        "url": url,
        "domain": _get_domain(url),
        "snippet": snippet,
        "content": None,
        "content_len": 0,
        "error": None,
    }
    try:
        fc = fetch_url(url, extract_mode="markdown", config=_FETCH_CFG)
        text = fc.get("text", "").strip()
        if len(text) >= _MIN_CONTENT_LEN:
            result["content"] = text
            result["content_len"] = len(text)
        else:
            result["error"] = f"nội dung quá ngắn ({len(text)} ký tự)"
    except Exception as e:
        result["error"] = str(e)
    return result


def _format_result(idx: int, r: dict) -> str:
    """Định dạng 1 kết quả thành text cho LLM."""
    header = f"[{idx}] {r['title']}  |  {r['domain']}\nURL: {r['url']}"
    if r["content"]:
        body = r["content"][:6_000]  # truncate thêm lần nữa khi format
        tag = f"({r['content_len']:,} ký tự)"
        return f"{header}  {tag}\n{body}"
    elif r["snippet"]:
        return f"{header}  (snippet only – fetch thất bại: {r['error']})\n{r['snippet']}"
    else:
        return f"{header}  (không có nội dung: {r['error']})"


def _searxng_search(query: str, k: int) -> list[dict]:
    """
    Gọi SearxNG với retry 2 lần nếu timeout.
    Trả về list dict {title, url, snippet} đã dedup theo URL.
    """
    # Health check trước khi search
    healthy, health_msg = _check_searxng_health()
    if not healthy:
        raise ConnectionError(f"SearXNG không sẵn sàng: {health_msg}")

    for attempt in range(2):
        try:
            raw = search_and_fetch(
                _SEARXNG_URL, query, top_k=k,
                extract_mode="markdown", config=_FETCH_CFG,
            )
            # search_and_fetch đã fetch content – chuyển sang format chuẩn
            results = []
            seen_urls: set[str] = set()
            for r in raw:
                url = r.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                fc = r.get("full_content")
                text = (fc.get("text", "") if fc else "").strip()
                results.append({
                    "title": r.get("title", ""),
                    "url": url,
                    "domain": _get_domain(url),
                    "snippet": r.get("snippet", ""),
                    "content": text if len(text) >= _MIN_CONTENT_LEN else None,
                    "content_len": len(text),
                    "error": r.get("error"),
                })
            return results
        except Exception as e:
            if attempt == 0:
                time.sleep(1.5)
            else:
                raise e
    return []


def _ddg_search(query: str, k: int) -> list[dict]:
    """
    DuckDuckGo fallback – fetch song song với ThreadPoolExecutor.
    """
    ddgs = DDGS()
    ddg_raw = list(ddgs.text(query, max_results=k + 2))  # lấy thêm dự phòng dedup

    # Dedup URL
    seen_urls: set[str] = set()
    unique_raw = []
    for r in ddg_raw:
        url = r.get("href", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_raw.append(r)
        if len(unique_raw) >= k:
            break

    # Fetch song song
    results: list[Optional[dict]] = [None] * len(unique_raw)
    with ThreadPoolExecutor(max_workers=5) as exe:
        futures = {
            exe.submit(
                _fetch_one_safe,
                r["href"],
                r.get("title", ""),
                r.get("body", ""),
            ): i
            for i, r in enumerate(unique_raw)
        }
        for future in as_completed(futures, timeout=20):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                r = unique_raw[idx]
                results[idx] = {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "domain": _get_domain(r.get("href", "")),
                    "snippet": r.get("body", ""),
                    "content": None,
                    "content_len": 0,
                    "error": str(e),
                }

    return [r for r in results if r is not None]


# ── Tool chính ────────────────────────────────────────────────────────────────
@tool
def web_search(query: str, k: int = WEB_SEARCH_TOP_K) -> str:
    """
    Tìm kiếm web và trả về nội dung đầy đủ của các trang liên quan.
    Dùng SearxNG (localhost:8888), tự động fallback DuckDuckGo nếu lỗi.

    Để kết quả tốt nhất:
      - Truyền từ khoá ngắn gọn, súc tích (tiếng Anh hoặc tiếng Việt đều được).
      - Tách nhiều chủ đề thành nhiều lần gọi web_search riêng.
      - Với câu hỏi thực tế/mới nhất, thêm năm hiện tại vào query.

    Args:
        query: Câu truy vấn tìm kiếm
        k:     Số kết quả tối đa (mặc định từ config, thường là 5)

    Returns:
        Chuỗi text chứa nội dung các trang, nguồn (domain), URL.
        Prefix "[SearxNG]" hoặc "[DuckDuckGo]" cho biết nguồn dữ liệu.
    """
    query = (query or "").strip()
    if not query:
        return "❌ Không có truy vấn tìm kiếm."

    source_label = "SearxNG"
    try:
        results = _searxng_search(query, k)
        if not results:
            raise ValueError("SearxNG trả về 0 kết quả")
    except Exception as searxng_err:
        source_label = "DuckDuckGo"
        try:
            results = _ddg_search(query, k)
        except Exception as ddg_err:
            return (
                f"❌ Tìm kiếm thất bại.\n"
                f"  SearxNG: {searxng_err}\n"
                f"  DuckDuckGo: {ddg_err}"
            )

    if not results:
        return f"[{source_label}] Không tìm thấy kết quả nào cho: {query}"

    # Sắp xếp: kết quả có full content lên trước
    results.sort(key=lambda r: r.get("content_len", 0), reverse=True)

    parts = [f"🔍 [{source_label}] Kết quả cho: «{query}»\n{'─'*60}"]
    for i, r in enumerate(results, 1):
        parts.append(_format_result(i, r))

    return "\n\n".join(parts)