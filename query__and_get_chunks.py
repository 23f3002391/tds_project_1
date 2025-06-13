import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

# Load index and metadata
def load_faiss_index(index_path: str, metadata_path: str):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Embed user query
def embed_query(query: str, model):
    return model.encode([query])[0]

# Search top-k results
def search_top_k(query_vector: np.ndarray, index, metadata, k=5):
    query_vector = np.array([query_vector]).astype("float32")
    distances, indices = index.search(query_vector, k)
    return [metadata[i] for i in indices[0] if i < len(metadata)]

# Format context for LLM
def format_context(chunks: List[dict]) -> str:
    return "\n\n---\n\n".join([f"Source: {chunk['url']}\n{chunk['chunk']}" for chunk in chunks])

# Example query function
def ask_question(query: str):
    print(f"\n🔎 Question: {query}")
    
    # Load everything
    index, metadata = load_faiss_index("conversations1.index", "conversations_metadata1.pkl")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Embed and search
    query_vec = embed_query(query, model)
    top_chunks = search_top_k(query_vec, index, metadata, k=10)
    
    # Format context
    context = format_context(top_chunks)

    # Send to LLM (pseudo — here you would send to Gemini/GPT)
    print("\n📄 Sending to LLM with context:\n")
    print(context)
    print("\n🧠 (Send the above + the question to Gemini, GPT, etc.)")

# Example usage
ask_question("The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?")
