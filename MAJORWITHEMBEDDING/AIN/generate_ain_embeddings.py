"""
Generate BGE-M3 embeddings for Ain chunked legal-document JSON files.

The embedding is generated from:
    title + summary

The embedding is then INSERTED DIRECTLY into the original chunk.

Model:
    BAAI/bge-m3

Dense embedding dimension:
    1024

Expected files:
    ch1_chunks_with_summaries.json
    ch2_chunks_with_summaries.json
    ...
    ch11_chunks_with_summaries.json

Output:
    chunks_with_embeddings.json
"""

import json
import glob
import torch
from FlagEmbedding import BGEM3FlagModel


# =========================================================
# 1. SETTINGS
# =========================================================

INPUT_PATTERN = "ch*_chunks_with_summaries.json"
OUTPUT_FILE = "chunks_with_embeddings.json"

# RTX 3050 4GB:
# Start with 4. If memory is sufficient, you can increase it.
BATCH_SIZE = 4

MAX_LENGTH = 512


# =========================================================
# 2. CHECK PYTORCH / GPU
# =========================================================

print("=" * 60)
print("AIN LEGAL DOCUMENT EMBEDDING GENERATION")
print("=" * 60)

print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA available: YES")
    print("GPU:", torch.cuda.get_device_name(0))
    USE_FP16 = True
else:
    print("CUDA available: NO")
    print("Using CPU")
    USE_FP16 = False


# =========================================================
# 3. LOAD BGE-M3 MODEL
# =========================================================

print("\nLoading BAAI/bge-m3...")

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=USE_FP16
)

print("Model loaded successfully.")


# =========================================================
# 4. FIND AIN CHUNK FILES
# =========================================================

chunk_files = sorted(
    glob.glob(INPUT_PATTERN)
)

if not chunk_files:
    raise FileNotFoundError(
        "No files matching "
        "'ch*_chunks_with_summaries.json' were found."
    )

print("\nAIN chunk files found:")

for path in chunk_files:
    print("  -", path)


# =========================================================
# 5. LOAD ALL CHUNKS
# =========================================================

all_chunks = []

for path in chunk_files:

    print(f"\nReading {path}...")

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError(
            f"{path} does not contain a JSON list of chunks."
        )

    print(f"  Loaded {len(chunks)} chunks.")

    all_chunks.extend(chunks)


print("\n" + "=" * 60)
print(f"TOTAL AIN CHUNKS: {len(all_chunks)}")
print("=" * 60)


if not all_chunks:
    raise ValueError(
        "No chunks were loaded."
    )


# =========================================================
# 6. BUILD TEXT FOR EMBEDDING
# =========================================================

texts = []

# Keep a reference to the ORIGINAL chunk objects.
# This allows us to insert the embedding directly
# into the original chunk later.

valid_chunks = []

for chunk in all_chunks:

    chunk_id = chunk.get("chunk_id")

    if not chunk_id:
        print(
            "Warning: skipping chunk without chunk_id."
        )
        continue

    title = (
        chunk.get("title") or ""
    ).strip()

    summary = (
        chunk.get("summary") or ""
    ).strip()


    # -----------------------------------------------------
    # Combine title + summary
    # -----------------------------------------------------

    if title and summary:

        combined = f"{title}. {summary}"

    elif title:

        combined = title

    elif summary:

        combined = summary

    else:

        print(
            f"Warning: skipping {chunk_id} "
            "because title and summary are empty."
        )

        continue


    texts.append(combined)

    # IMPORTANT:
    # Store the ORIGINAL chunk object.
    valid_chunks.append(chunk)


print(
    f"\nPrepared {len(texts)} texts for embedding."
)


if not texts:

    raise ValueError(
        "No valid title/summary texts were found."
    )


# =========================================================
# 7. SHOW SAMPLE EMBEDDING TEXT
# =========================================================

print("\nSample text being embedded:")
print("-" * 60)

print(
    texts[0][:1000]
)

print("-" * 60)


# =========================================================
# 8. GENERATE BGE-M3 DENSE EMBEDDINGS
# =========================================================

print("\nGenerating embeddings...")

output = model.encode(
    texts,

    batch_size=BATCH_SIZE,

    max_length=MAX_LENGTH,

    return_dense=True,

    return_sparse=False,

    return_colbert_vecs=False,
)


dense_vecs = output["dense_vecs"]


print(
    f"Generated {len(dense_vecs)} embeddings."
)

print(
    "Embedding shape:",
    dense_vecs.shape
)


# =========================================================
# 9. INSERT EMBEDDING INTO ORIGINAL CHUNKS
# =========================================================

print(
    "\nAdding embeddings directly "
    "to the original AIN chunks..."
)

for chunk, vector in zip(
    valid_chunks,
    dense_vecs
):

    # This DOES NOT replace the chunk.
    #
    # It adds a new field called "embedding"
    # while keeping every existing field.

    chunk["embedding"] = vector.tolist()


print(
    "Embeddings successfully attached "
    "to original chunks."
)


# =========================================================
# 10. VERIFY RESULT
# =========================================================

print("\n" + "=" * 60)
print("VERIFYING FIRST CHUNK")
print("=" * 60)

first_chunk = valid_chunks[0]

print("\nChunk ID:")
print(
    first_chunk.get("chunk_id")
)

print("\nFields in first chunk:")

for key in first_chunk.keys():

    print(
        "  -",
        key
    )


print("\nEmbedding exists:")

print(
    "embedding" in first_chunk
)


if "embedding" in first_chunk:

    print(
        "Embedding dimension:",
        len(first_chunk["embedding"])
    )

    print(
        "First 5 values:",
        first_chunk["embedding"][:5]
    )


# =========================================================
# 11. SAVE COMPLETE AIN CHUNKS
# =========================================================

print(
    "\nSaving complete chunks..."
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_chunks,
        f,
        ensure_ascii=False,
        indent=2
    )


# =========================================================
# 12. FINAL VERIFICATION
# =========================================================

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print(
    "Total original chunks:",
    len(all_chunks)
)

print(
    "Chunks embedded:",
    len(valid_chunks)
)

print(
    "Embedding dimension:",
    len(valid_chunks[0]["embedding"])
)

print(
    "Output file:",
    OUTPUT_FILE
)

print("\nEach embedded AIN chunk now contains:")

print("  ✓ Original chunk_id")
print("  ✓ Original chapter information")
print("  ✓ Original clause/subclause information")
print("  ✓ Original title")
print("  ✓ Original content")
print("  ✓ Original summary")
print("  ✓ 1024-dimensional BGE-M3 embedding")

print("\nDone!")