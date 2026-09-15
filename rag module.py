import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class BookRAGAgent:
    """
    Modular, CPU-friendly Hybrid RAG Agent (BM25 + Gemini Embeddings via RRF)
    with strict 'INSUFFICIENT_CONTEXT' guardrails.
    """
    def __init__(self, pdf_path: str, chunk_size: int = 600, chunk_overlap: int = 200):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: '{pdf_path}'")

        # 1. Load & Chunk Document
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(docs)

        # 2. Local Keyword Retriever (BM25)
        self.bm25 = BM25Retriever.from_documents(chunks)
        self.bm25.k = 4

        # 3. Cloud Vector Retriever (Gemini Embeddings - 0% CPU)
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        self.vector = vectorstore.as_retriever(search_kwargs={"k": 4})

        # 4. LLM & Prompt Setup
        self.llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
        self.prompt = ChatPromptTemplate.from_template(
            """You are a strict document QA assistant. Answer ONLY using the provided book context below.
Do not use outside knowledge or make assumptions.

If the provided context does not contain enough information to answer the question thoroughly, reply EXACTLY with:
"INSUFFICIENT_CONTEXT: the requested information is not present in the book".

Context (Hybrid BM25 + Semantic Vector Search Results):
{context}

Question: {question}"""
        )

    def _rrf_search(self, query: str, k: int = 4):
        bm25_docs = self.bm25.invoke(query)
        vector_docs = self.vector.invoke(query)
        
        scores = {}
        c = 60
        for rank, doc in enumerate(bm25_docs):
            scores[doc.page_content] = scores.get(doc.page_content, 0) + (0.5 / (c + rank + 1))
        for rank, doc in enumerate(vector_docs):
            scores[doc.page_content] = scores.get(doc.page_content, 0) + (0.5 / (c + rank + 1))

        sorted_text = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        all_docs = bm25_docs + vector_docs
        
        return [next(doc for doc in all_docs if doc.page_content == txt) for txt, _ in sorted_text]

    def ask(self, question: str) -> str:
        retrieved_docs = self._rrf_search(question)
        context = "\n---\n".join(doc.page_content for doc in retrieved_docs)
        chain = self.prompt | self.llm
        response = chain.invoke({"context": context, "question": question})
        return str(response.content)

# Direct execution entry point
if __name__ == "__main__":
    print("--- Hybrid Book RAG Agent ---")
    pdf_input = input("Enter PDF file name or path (e.g., sample.pdf): ").strip().strip('"')
    pdf_path = pdf_input if pdf_input else "sample.pdf"

    agent = BookRAGAgent(pdf_path)
    print(f"\nSuccessfully loaded and indexed '{pdf_path}'. Ready for questions! Type 'exit' to quit.")

    while True:
        user_question = input("\nYour Question: ").strip()

        if user_question.lower() in ["exit", "quit"]:
            print("Exiting RAG agent.")
            break
        elif not user_question:
            continue

        answer = agent.ask(user_question)
        print(f"\nAgent Answer:\n{answer}")
