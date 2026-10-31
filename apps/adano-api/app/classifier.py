from functools import lru_cache

import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=4096)
def _embed_cached(text: str) -> bytes:
    return _model().encode(text, normalize_embeddings=True).astype(np.float32).tobytes()


def embed(text: str) -> np.ndarray:
    return np.frombuffer(_embed_cached(text), dtype=np.float32)


def warmup() -> None:
    embed("warmup")


def softmax(x, T: float = 0.05) -> np.ndarray:
    z = np.array(x, dtype=np.float64) / T
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def category_text(c: dict) -> str:
    desc = (c.get("description") or "").strip()
    return f"{c['name']}: {desc}" if desc else c["name"]


def classify(
    text: str,
    categories: list[dict],
    examples: dict[int, list[np.ndarray]],
    N0: int = 10,
    k: int = 3,
    tau: float = 0.5,
    margin: float = 0.1,
) -> tuple[int | None, float]:
    """Hybrid scorer (thesis 3.6).

    categories: [{id, name, description}]
    examples: category_id -> embeddings of user-confirmed notes
    Returns (category_id | None, confidence). None = Uncategorized.
    """
    if not categories:
        return None, 0.0

    x = embed(text)
    scores = []
    for c in categories:
        sim_desc = float(x @ embed(category_text(c)))
        ex_vecs = examples.get(c["id"], [])
        n = len(ex_vecs)
        if n:
            ex = np.stack(ex_vecs)
            sims = np.sort(ex @ x)[::-1][: min(k, n)]
            sim_ex = float(sims.mean())
            alpha = min(1.0, n / N0)
        else:
            sim_ex, alpha = 0.0, 0.0
        scores.append((1 - alpha) * sim_desc + alpha * sim_ex)

    probs = softmax(scores)
    order = np.argsort(probs)[::-1]
    top1 = float(probs[order[0]])
    top2 = float(probs[order[1]]) if len(order) > 1 else 0.0

    if top1 >= tau and (top1 - top2) >= margin:
        return categories[int(order[0])]["id"], top1
    return None, top1
