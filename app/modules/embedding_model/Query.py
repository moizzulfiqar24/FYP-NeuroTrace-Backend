import os
import faiss
import pickle
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer
import openai

# ======================== CONFIG ========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../embedding_model/Speech To Text Model/tiny.en.pt"))
BASE_PATH = os.path.abspath(os.path.join(BASE_DIR, "../DB"))
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Groq API setup
GROQ_API_KEY = "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"
GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama3-8b-8192"

client = openai.OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_API_BASE
)

# ======================== FUNCTIONS ========================

def query_vector_store(user_id: str, query: str, top_k: int = 1) -> list[dict]:
    """
    Query the user's FAISS vector store for top matching chunks,
    then use Groq LLM to generate a final answer, and add it to the result.

    Args:
        user_id (str): UUID of the user.
        query (str): Query string.
        top_k (int): Number of top results to return.

    Returns:
        list of dicts, each with 'text' and 'timestamp', plus one extra with LLM 'text' and 'timestamp'.
    """
    # Validate UUID format
    try:
        uuid.UUID(user_id)
    except ValueError:
        raise ValueError("Invalid UUID format provided for user_id.")

    vector_path = os.path.join(BASE_PATH, "VectorStore", str(user_id))
    index_file = os.path.join(vector_path, "chunk_index.faiss")
    meta_file = os.path.join(vector_path, "chunk_metadata.pkl")

    if not os.path.exists(index_file) or not os.path.exists(meta_file):
        raise FileNotFoundError(f"❌ Vector DB files not found for user {user_id}.")

    print("📥 Loading vector index and metadata...")
    index = faiss.read_index(index_file)

    with open(meta_file, "rb") as f:
        metadata = pickle.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL)
    query_vec = model.encode([query])[0].reshape(1, -1)

    # Perform FAISS search
    _, indices = index.search(query_vec, top_k)

    # Prepare result list
    results = []
    context_chunks = []

    for i in indices[0]:
        item = metadata[i]
        timestamp = datetime.fromisoformat(item["timestamp"])
        results.append({
            "text": item["text"],
            "timestamp": timestamp
        })
        context_chunks.append(item["text"])

    # Step: If no chunks found, return empty list
    if not context_chunks:
        return results

    # Step: Build context and call Groq LLM
    context = "\n".join(context_chunks)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use ONLY the provided context to answer the question. Be concise and accurate, no more than 2–3 sentences."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2
    )

    final_answer = response.choices[0].message.content.strip()

    # Step: Add LLM's answer as an extra result
    results.append({
        "text": final_answer,
        "timestamp": datetime.now()  # current timestamp
    })

    return results
