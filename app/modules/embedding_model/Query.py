import os
import faiss
import pickle
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer
import openai
from elasticsearch import Elasticsearch, exceptions as es_exceptions

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

# ElasticSearch setup
ES_HOST = "http://localhost:9200"

es = Elasticsearch(
    ES_HOST,
    headers={
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
    }
)

# Sentence Transformer for embeddings
model = SentenceTransformer(EMBEDDING_MODEL)

# ======================== FUNCTIONS ========================

def create_index_if_not_exists(user_index_name: str, metadata: list[dict]):
    """
    Create a user-specific ElasticSearch index if it doesn't exist,
    and upload the chunks from metadata.
    """
    try:
        if not es.indices.exists(index=user_index_name):
            print(f"🛠 Creating Elastic index for user: {user_index_name}")
            es.indices.create(
                index=user_index_name,
                body={
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "timestamp": {"type": "date"}
                        }
                    }
                }
            )

            # Upload existing FAISS metadata into this index
            for item in metadata:
                doc = {
                    "text": item["text"],
                    "timestamp": item["timestamp"]
                }
                es.index(index=user_index_name, document=doc)

            print(f"✅ Uploaded {len(metadata)} chunks into {user_index_name} index.")
        else:
            print(f"ℹ️ Elastic index {user_index_name} already exists.")
    except es_exceptions.ElasticsearchException as e:
        print(f"❌ Failed to create or upload ElasticSearch index: {e}")
        raise

def query_vector_store(user_id: str, query: str, top_k: int = 1) -> list[dict]:
    """
    Query the user's FAISS vector store and ElasticSearch index,
    then use Groq LLM to generate a final answer, and add it to the result.
    """
    # Validate UUID
    try:
        uuid.UUID(user_id)
    except ValueError:
        raise ValueError("Invalid UUID format provided for user_id.")

    user_index_name = f"{user_id}_index"
    vector_path = os.path.join(BASE_PATH, "VectorStore", str(user_id))
    index_file = os.path.join(vector_path, "chunk_index.faiss")
    meta_file = os.path.join(vector_path, "chunk_metadata.pkl")

    if not os.path.exists(index_file) or not os.path.exists(meta_file):
        raise FileNotFoundError(f"❌ Vector DB files not found for user {user_id}.")

    print(f"📥 Loading FAISS index and metadata for user: {user_id}...")
    index = faiss.read_index(index_file)

    with open(meta_file, "rb") as f:
        metadata = pickle.load(f)

    # Create ElasticSearch index if missing
    create_index_if_not_exists(user_index_name, metadata)

    query_vec = model.encode([query])[0].reshape(1, -1)

    # ======================== FAISS retrieval ========================

    print("🔎 Searching FAISS vector store...")
    _, indices = index.search(query_vec, top_k)

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

    # ======================== ElasticSearch retrieval ========================

    print("🔍 Searching ElasticSearch index...")
    es_query = {
        "query": {
            "match": {
                "text": query
            }
        },
        "size": top_k
    }

    es_response = es.search(index=user_index_name, body=es_query)

    for hit in es_response["hits"]["hits"]:
        text = hit["_source"].get("text", "")
        timestamp_str = hit["_source"].get("timestamp", datetime.now().isoformat())
        timestamp = datetime.fromisoformat(timestamp_str)

        results.append({
            "text": text,
            "timestamp": timestamp
        })
        context_chunks.append(text)

    # ======================== Final Step: Call LLM ========================

    if not context_chunks:
        return results

    print("🤖 Calling Groq LLM for final answer...")
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

    results.append({
        "text": final_answer,
        "timestamp": datetime.now()
    })

    return results
