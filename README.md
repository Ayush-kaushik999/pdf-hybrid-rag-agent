# CPU-Friendly Hybrid RAG Agent (BM25 + Gemini Embeddings)

A lightweight, production-grade Hybrid Retrieval-Augmented Generation (RAG) agent designed for resource-constrained environments (e.g., dual-core/quad-core laptops without dedicated GPUs). 

By delegating heavy vector calculations to **Google's Gemini Cloud Embeddings** and combining them with a fast local **BM25 keyword retriever**, this system delivers hybrid search precision while keeping local CPU and RAM usage near zero. Inference is powered by Groq's `openai/gpt-oss-120b` model with strict document QA guardrails.

---

## Key Features

* **Hybrid Retrieval (BM25 + Semantic Vector Search):** Combines exact keyword matching (BM25) with high-dimensional cloud embeddings for superior contextual accuracy.
* **Zero Local Vector Compute:** Offloads dense embedding generation to `gemini-embedding-001` via Google GenAI API, bypassing local model overhead.
* **Custom Reciprocal Rank Fusion (RRF):** Implements an explicit RRF rank-merging algorithm to combine keyword and vector results cleanly without version-locked framework wrappers.
* **Strict QA Guardrails:** Configured to strictly suppress hallucinations. Out-of-document queries safely trigger a structured `INSUFFICIENT_CONTEXT` fallback response.
* **Modular Dual-Purpose Architecture:** Written as a clean `BookRAGAgent` class (`rag_module.py`) that functions both as an interactive CLI application and as an importable python module.

---

## Tech Stack & Architecture

* **Language:** Python 3.13
* **Framework:** LangChain (Core & Community)
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Embeddings API:** Google Generative AI (`gemini-embedding-001`)
* **Keyword Search:** `rank_bm25` / `BM25Retriever`
* **LLM Engine:** Groq API (`openai/gpt-oss-120b`)
* **Document Parsing:** `PyPDFLoader`

---

## Project Structure

```text
pdf-hybrid-rag-agent/
├── rag_module.py       # Core BookRAGAgent class & CLI entry point
├── requirements.txt    # Pinned dependency definitions
├── .gitignore          # Environment and binary file protections
├── .env.example        # Environment variable template
└── README.md           # Project documentation
```

---

## Getting Started

### Prerequisites

* Python 3.10+ (Tested on Python 3.13)
* A [Groq API Key](https://console.groq.com/)
* A [Google Gemini API Key](https://aistudio.google.com/)

---

### Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Ayush-kaushik999/pdf-hybrid-rag-agent.git
   cd pdf-hybrid-rag-agent
   ```

2. **Create and Activate a Virtual Environment:**
   * **Windows (PowerShell / Command Prompt):**
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   * **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key_here
   GOOGLE_API_KEY=AIzaSy_your_google_api_key_here
   ```

---

## Usage

### 1. Direct Execution via CLI

Run `rag_module.py` directly to start an interactive Q&A session:

```bash
python rag_module.py
```

1. Enter the path or file name of your target PDF document when prompted (e.g., `sample.pdf`).
2. Ask questions directly in the terminal interface.
3. Type `exit` or `quit` to end the session.

### 2. Importing into External Python Projects

You can import `BookRAGAgent` directly as a standalone module in your own scripts or application backends:

```python
from rag_module import BookRAGAgent

# Initialize and index PDF document
agent = BookRAGAgent("path/to/your_document.pdf")

# Execute query
answer = agent.ask("What are the key findings presented in Chapter 3?")
print(answer)
```

---

## Retrieval Mechanics: Custom RRF Fusion

Standard RAG architectures often rely on pure vector distance, missing critical technical jargon or exact serial numbers. This agent uses **Reciprocal Rank Fusion (RRF)** to combine ranked document streams from BM25 and FAISS:

$$RRF\_Score(d \in D) = \sum_{m \in M} rac{w_m}{k + r_m(d)}$$

Where:
* $M$ represents the search methods (BM25 Keyword and Gemini Vector).
* $r_m(d)$ is the 1-based rank position of document $d$ within retriever $m$.
* $k$ is a smoothing constant set to $60$.
* $w_m$ distributes equal weight ($0.5$) across keyword and semantic channels.

---

## License

Distributed under the MIT License. See `LICENSE` for details.
