import math
import os
import sys
import pypdf
import requests
from dotenv import load_dotenv

# 1. Environment & Config Setup
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PDF_PATH = "sample.pdf"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-120b"  # Active Groq model


# 2. PDF Parsing & Text Chunking
def load_pdf_chunks(filepath: str, chunk_size=500, overlap=100) -> list[str]:
    if not os.path.exists(filepath):
        print(f"[Error] '{filepath}' not found.")
        sys.exit(1)

    reader = pypdf.PdfReader(filepath)
    full_text = "\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )

    chunks, start = [], 0
    while start < len(full_text):
        chunks.append(full_text[start : start + chunk_size].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


# 3. Vectorization & Cosine Similarity Engine
def get_term_counts(text: str) -> dict[str, int]:
    words = text.lower().split()
    counts = {}
    for w in words:
        clean_w = w.strip(".,!?\"'()[]{}")
        if clean_w:
            counts[clean_w] = counts.get(clean_w, 0) + 1
    return counts


def calculate_cosine_sim(vec1: dict, vec2: dict) -> float:
    if not vec1 or not vec2:
        return 0.0
    dot_product = sum(vec1[w] * vec2[w] for w in set(vec1) & set(vec2))
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return dot_product / (mag1 * mag2) if (mag1 * mag2) > 0 else 0.0


def retrieve_relevant_chunks(
    query: str, chunks: list[str], top_k=2
) -> list[str]:
    q_vec = get_term_counts(query)
    scored = [(calculate_cosine_sim(q_vec, get_term_counts(c)), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]


# 4. API Invocation & RAG Execution
def ask_rag(query: str, chunks: list[str]) -> str:
    matched_context = "\n---\n".join(retrieve_relevant_chunks(query, chunks))
    prompt = (
        "You are a strict document QA bot. Answer ONLY using the context.\n"
        "If the answer is not in the context, reply EXACTLY 'INSUFFICIENT_CONTEXT'.\n\n"
        f"Context:\n{matched_context}\n\nQuestion: {query}"
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }

    try:
        res = requests.post(GROQ_URL, headers=headers, json=payload, timeout=10)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as err:
        return f"[API Error]: {err}"


# 5. Main Execution Loop
if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("[Error] GROQ_API_KEY missing in .env")
        sys.exit(1)

    print(f"Loading '{PDF_PATH}'...")
    pdf_chunks = load_pdf_chunks(PDF_PATH)
    print(f"RAG Ready! Split into {len(pdf_chunks)} chunks.")

    while True:
        user_input = input("\nYour Question: ").strip()
        match user_input.lower():
            case "exit" | "quit":
                print("Exiting Manual RAG Session.")
                break
            case "":
                continue
            case _:
                print("Retrieving context & querying Cloud LLM...")
                answer = ask_rag(user_input, pdf_chunks)
                print(f"\nAgent Answer:\n{answer}")
