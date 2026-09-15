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

pdf_input = input("enter pdf file name or file path (e.g. , my_book.pdf) :").strip()
PDF_FILE=pdf_input if pdf_input else"sample.pdf"
if not os.path.exists(PDF_FILE):
    raise FileNotFoundError(f"'{PDF_FILE}' missing! Create it first.")
print(f"loading and indexing '{PDF_FILE}'...")

# 1. Document Loading & Chunking
loader = PyPDFLoader(PDF_FILE)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# 2. Local BM25 Retriever (Keyword matching)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

# 3. Cloud Semantic Vector Retriever (Zero CPU load on your laptop)
embeddings=GoogleGenerativeAIEmbeddings(model= "gemini-embedding-001")
vectorstore=FAISS.from_documents(chunks,embeddings)
vector_retriever=vectorstore.as_retriever(search_kwargs={"k":4})

# 4. Custom Reciprocal Rank Fusion (Replaces EnsembleRetriever)
def hybrid_combine_docs(query: str, k: int = 4):
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)
    
    # Reciprocal Rank Fusion (RRF) scoring algorithm
    doc_scores = {}
    c = 60  # RRF Constant
    
    for rank, doc in enumerate(bm25_docs):
        content = doc.page_content
        doc_scores[content] = doc_scores.get(content, 0) + (0.5 / (c + rank + 1))
        
    for rank, doc in enumerate(vector_docs):
        content = doc.page_content
        doc_scores[content] = doc_scores.get(content, 0) + (0.5 / (c + rank + 1))
        
    # Sort docs by highest combined score
    sorted_contents = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    # Reconstruct document objects
    combined_docs = []
    all_retrieved = bm25_docs + vector_docs
    for content, _ in sorted_contents:
        for doc in all_retrieved:
            if doc.page_content == content:
                combined_docs.append(doc)
                break
                
    return combined_docs


# 5. Cloud LLM (Groq handles all computation)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)

#prompts:

qa_prompt = ChatPromptTemplate.from_template(
    """You are a strict document QA assistant. Answer ONLY using the provided book context below.
    Do not use outside knowledge or make assumptions.

If the provided context does not contain enough information to answer the question thoroughly , reply EXACTLY with :
"INSUFFICIENT_CONTEXT: the requested information is not present in the book".

Context(Hybrid BM25 + Semantic Vector Search Results):
{context}

Question: {question}"""
)

def format_docs(retrieved_docs):
    return "\n---\n".join(doc.page_content for doc in retrieved_docs)

def run_hybrid_rag(question: str)-> str:
    retrieved_docs= hybrid_combine_docs(question)
    formatted_context= format_docs(retrieved_docs)
    chain=qa_prompt | llm
    response=chain.invoke({"context": formatted_context, "question": question})

    #ensure standard string return type
    return str(response.content)
 
if __name__ == "__main__":
    
    print("Welcome to Hybrid book-only RAG Agent!")

    while True:
        user_question = input("\nYour Question: ").strip()

        match user_question.lower():
            case "exit" | "quit":
                print("Exiting RAG agent.")
                break
            case "":
                continue
            case _:
                ans = run_hybrid_rag(user_question)
                print(f"\nAgent Answer:\n{ans}")