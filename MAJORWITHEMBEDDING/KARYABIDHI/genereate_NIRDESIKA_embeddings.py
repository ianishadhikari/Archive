"""
Generate embeddings for title + summary fields from chunked legal-document JSON files.
Model: BAAI/bge-m3 (open source, MIT license, strong Devanagari/Nepali support)

Install first (on your own machine, not this sandbox):
    pip install FlagEmbedding torch

Usage:
    python generate_embeddings.py
"""

import json
import glob
import numpy as np
from FlagEmbedding import BGEM3FlagModel

# ---- 1. Load model ----
# use_fp16=True speeds things up on GPU; set False if running CPU-only and it errors
import torch

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=torch.cuda.is_available()
)

# ---- 2. Load all chunk files ----
all_chunks = []
for path in sorted(glob.glob("ch*_chunks_with_summaries.json")):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        all_chunks.extend(chunks)

print(f"Loaded {len(all_chunks)} chunks total")

# ---- 3. Build text to embed: title + summary ----
texts = []
for c in all_chunks:
    title = (c.get("title") or "").strip()
    summary = (c.get("summary") or "").strip()
    combined = f"{title}. {summary}".strip()
    texts.append(combined)

# ---- 4. Generate embeddings (dense vectors) ----
# BGE-M3 also supports sparse + multi-vector (colbert) outputs if you want hybrid search later
output = model.encode(
    texts,
    batch_size=12,
    max_length=512,          # title+summary is short, 512 is plenty
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False,
)

dense_vecs = output["dense_vecs"]  # shape: (num_chunks, 1024)

# ---- 5. Save embeddings separately ----

embedding_records = []

for c, vec in zip(all_chunks, dense_vecs):
    embedding_records.append({
        "chunk_id": c["chunk_id"],
        "embedding": vec.tolist()
    })

with open("chunk_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embedding_records, f, ensure_ascii=False, indent=2)

print(f"Saved {len(embedding_records)} embeddings.")
print("Output file: chunk_embeddings.json")
print("Embedding dimension:", len(dense_vecs[0]))