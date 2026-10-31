"""
Generate embeddings for title + summary fields from the summarized legal-document JSON chunks.

Model: BAAI/bge-m3

Install first:
    pip install FlagEmbedding torch numpy

Usage:
    python generate_core_embeddings.py

Expected input files:
    ch1_chunks_with_summaries.json
    ch2_chunks_with_summaries.json
    ...
    ch10_chunks_with_summaries.json

Output:
    chunk_embeddings.json
"""

import json
import glob
import torch
from FlagEmbedding import BGEM3FlagModel


# ---- 1. Load embedding model ----
model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=torch.cuda.is_available()
)


# ---- 2. Load chapter chunk files ----
# This intentionally loads only ch*_chunks_with_summaries.json
# so all_chunks_with_summaries.json is not loaded as well and duplicated.
all_chunks = []

for path in sorted(glob.glob("ch*_chunks_with_summaries.json")):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        all_chunks.extend(chunks)

if not all_chunks:
    raise FileNotFoundError(
        "No files matching 'ch*_chunks_with_summaries.json' were found."
    )

print(f"Loaded {len(all_chunks)} chunks total")


# ---- 3. Build embedding text: title + summary ----
texts = []

for chunk in all_chunks:
    title = (chunk.get("title") or "").strip()
    summary = (chunk.get("summary") or "").strip()

    if title and summary:
        combined = f"{title}. {summary}"
    elif title:
        combined = title
    elif summary:
        combined = summary
    else:
        combined = ""

    texts.append(combined)


# ---- 4. Generate dense embeddings ----
output = model.encode(
    texts,
    batch_size=12,
    max_length=512,
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False,
)

dense_vecs = output["dense_vecs"]


# ---- 5. Save embeddings ----
embedding_records = []

for chunk, vec in zip(all_chunks, dense_vecs):
    embedding_records.append({
        "chunk_id": chunk["chunk_id"],
        "embedding": vec.tolist()
    })

with open("chunk_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(
        embedding_records,
        f,
        ensure_ascii=False,
        indent=2
    )


print(f"Saved {len(embedding_records)} embeddings.")
print("Output file: chunk_embeddings.json")
print("Embedding dimension:", len(dense_vecs[0]))
