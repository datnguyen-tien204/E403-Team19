"""
rag/reranker.py
Lightweight cross-encoder reranker for RAG results.
"""
import torch
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_model():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_id = "BAAI/bge-reranker-v2-m3"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model = model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        print(f"[Reranker] load failed: {e}")
        return None, None, "cpu"


def rerank(query: str, passages: list[str], top_k: int = 4) -> list[tuple[str, float]]:
    """
    Score and return top_k passages with descending scores.
    """
    if not passages:
        return []
    tokenizer, model, device = _get_model()
    if model is None:
        return [(p, 0.0) for p in passages[:top_k]]

    pairs = [[query, p] for p in passages]
    with torch.no_grad():
        batch = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(device)
        scores = model(**batch).logits.view(-1)
        scores = scores.float().cpu().tolist()

    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

