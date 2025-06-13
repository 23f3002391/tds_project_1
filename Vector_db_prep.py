import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

# Load conversations JSON
def load_conversations(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Flatten and chunk all posts from a conversation
def chunk_conversations(conversations, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks, metadata = [], []

    for convo in conversations:
        url = convo["url"]
        all_text = "\n\n".join(convo["posts"])
        convo_chunks = splitter.split_text(all_text)

        for chunk in convo_chunks:
            chunks.append(chunk)
            metadata.append({
                "url": url,
                "chunk": chunk
            })

    return chunks, metadata

# Embed all chunks
def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(chunks, show_progress_bar=True)

# Save to FAISS + metadata file
def save_index(vectors, metadata, index_path, metadata_path):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

def run_pipeline():
    print("🔄 Loading conversations...")
    conversations = load_conversations("all_conversations1.json")

    print("✂️ Chunking posts...")
    chunks, metadata = chunk_conversations(conversations)

    print(f"🔢 Embedding {len(chunks)} chunks...")
    vectors = embed_chunks(chunks)

    print("💾 Saving to FAISS and metadata file...")
    save_index(np.array(vectors), metadata, "conversations1.index", "conversations_metadata1.pkl")

    print("✅ Done. Vector DB ready!")

if __name__ == "__main__":
    run_pipeline()
