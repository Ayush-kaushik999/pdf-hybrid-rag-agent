from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("sample.pdf", pagesize=letter)
styles = getSampleStyleSheet()

content = [
    Paragraph("**Artificial Intelligence and RAG Systems**", styles['Heading1']),
    Spacer(1, 12),
    Paragraph("Retrieval-Augmented Generation (RAG) combines document retrieval with Large Language Models.", styles['Normal']),
    Spacer(1, 12),
    Paragraph("Key models included in this test ecosystem are Groq Llama 3.3 70B and HuggingFace MiniLM embeddings.", styles['Normal']),
]

doc.build(content)
print("Successfully created a valid sample.pdf!")
