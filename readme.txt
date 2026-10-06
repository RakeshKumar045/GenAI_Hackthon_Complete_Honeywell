RAG Pipeline Coding Challenge (90 Minutes)

Welcome to the AI Engineer II – Coding / Hackathon Round.
Your task is to build a Retrieval-Augmented Generation (RAG) pipeline end-to-end using Python only.

You will be provided with:
•	A multi-page PDF file (contains text + tables).
•	A set of queries that must be answered only using information from the PDF.

Your goal is to implement a working RAG system that:
1.	Extracts information from the PDF
2.	Creates embeddings & builds a vector store
3.	Retrieves relevant chunks for a query
4.	Uses an open-source LLM for generation
5.	Produces accurate answers grounded in the document
________________________________________
✅ Problem Description

You are required to build a complete RAG pipeline that performs the following steps:
________________________________________
1. PDF Data Extraction

Extract all meaningful content from the PDF, including:
•	Page text
•	Tables / grid structures

You may use any Python PDF library (e.g., pdfplumber, PyMuPDF, etc.).
________________________________________
2. Text Chunking

Split extracted content into smaller chunks.

Requirements:
•	Fixed-size chunks (can be token-based or word-based)
•	Some overlap is recommended
•	Keep track of metadata (page number, chunk id, etc.)
________________________________________
3. Embedding Generation

Convert all chunks into vector embeddings using an open-source embedding model, such as:
•	sentence-transformers/all-MiniLM-L6-v2
•	Or any similar free model from HuggingFace
________________________________________
4. Vector Database

Store embeddings in an efficient vector index such as:
•	FAISS (recommended)
•	Annoy / HNSWlib / custom cosine similarity index

The index should allow top-K similarity search.
________________________________________
5. Retrieval for a Query

For each query:
•	Embed the query
•	Retrieve top relevant chunks
•	Build a context prompt
________________________________________
6. LLM-based Answer Generation

Use a free & open-source model such as:
•	google/flan-t5-small
•	tiiuae/falcon-rw-1b
•	Or any model available on HuggingFace without paid API keys

Your answer must be supported by the retrieved context from the PDF.
If the answer is not present in the document, respond with:

“I don’t know from the document.”
________________________________________
7. Final Output

Your final output must be a JSON file containing answers for each query:

[
{
"query": "...",
"answer": "...",
"retrieved_chunks": [
{"page": 1, "score": 0.81},
{"page": 3, "score": 0.76}
]
}
]

Submit the following :
1) Python Code (.py or .ipynb)
2) A json file with ur generated answers