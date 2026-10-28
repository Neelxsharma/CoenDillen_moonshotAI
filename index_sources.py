from pathlib import Path
import hashlib

import chromadb
import ollama

ROOT = Path(__file__).parent
SOURCES = ROOT / "sources"
DATABASE = ROOT / "data" / "chroma"
COLLECTION_NAME = "coen_history"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"


def split_into_chunks(text, size=1200, overlap=150):
    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


client = chromadb.PersistentClient(path=str(DATABASE))
collection = client.get_or_create_collection(COLLECTION_NAME)

files = list(SOURCES.glob("*.md")) + list(SOURCES.glob("*.txt"))

if not files:
    raise SystemExit("No .md or .txt source notes found in the sources folder.")

# Remove indexed records for files that have been moved out of sources.
current_source_names = {file_path.name for file_path in files}
existing = collection.get(include=["metadatas"])
indexed_source_names = {
    metadata.get("source")
    for metadata in existing["metadatas"]
    if metadata and metadata.get("source")
}
for stale_source in indexed_source_names - current_source_names:
    collection.delete(where={"source": stale_source})
    print(f"Removed old indexed material for {stale_source}")

for file_path in files:
    text = file_path.read_text(encoding="utf-8")
    chunks = split_into_chunks(text)

    # Replace this file's older indexed material if you run the script again.
    collection.delete(where={"source": file_path.name})

    if not chunks:
        continue

    embeddings = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunks,
    ).embeddings

    ids = [
        hashlib.sha256(
            f"{file_path.name}:{index}:{chunk}".encode("utf-8")
        ).hexdigest()
        for index, chunk in enumerate(chunks)
    ]

    metadata = [{"source": file_path.name} for _ in chunks]

    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadata,
        embeddings=embeddings,
    )

    print(f"Indexed {len(chunks)} section(s) from {file_path.name}")

print(f"Database now contains {collection.count()} section(s).")