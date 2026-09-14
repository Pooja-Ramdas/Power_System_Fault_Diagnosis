"""
Local, offline Spanish -> English translation using MarianMT (Helsinki-NLP
opus-mt-es-en). No API key, no per-call cost, runs on CPU (slow-ish, fine for
a few hundred/thousand short reports). Model weights (~300MB) are cached
once in ~/.cache/huggingface and are NOT part of your dataset's disk budget.
"""
from functools import lru_cache

_MODEL_NAME = "Helsinki-NLP/opus-mt-es-en"


@lru_cache(maxsize=1)
def _load_model(model_name=_MODEL_NAME):
    from transformers import MarianMTModel, MarianTokenizer
    tok = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tok, model


def translate_es_to_en(text, max_chunk_chars=1200, model_name=_MODEL_NAME):
    """Chunks long report text (MarianMT has a short effective context) and
    translates all chunks in a single batched inference call for high throughput."""
    text = (text or "").strip()
    if not text:
        return ""
    tok, model = _load_model(model_name)
    chunks = [text[i:i + max_chunk_chars] for i in range(0, len(text), max_chunk_chars)]
    if not chunks:
        return ""
    batch = tok(chunks, return_tensors="pt", truncation=True, padding=True)
    gen = model.generate(**batch, max_length=512)
    decoded = tok.batch_decode(gen, skip_special_tokens=True)
    return " ".join(decoded)

