from fastapi import FastAPI
from pydantic import BaseModel



from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


from dotenv import load_dotenv
import os
import json

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import pipeline


app = FastAPI(title="RAG API")



class UserQuery(BaseModel):
    query: str



embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



vector_db = FAISS.load_local(
    "vector_db",
    embedding_model,
    allow_dangerous_deserialization=True
)

load_dotenv()

print("create llm")
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    
    temperature=0.3
)


model_name = "google/flan-t5-small"


opensource_llm = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)


def retrieve(query, top_k=3):

    docs = vector_db.similarity_search_with_score(
        query,
        k=top_k
    )

    return docs


def build_context(query):

    retrieved_docs = retrieve(query)

    context = ""

    retrieved_chunks = []

    for doc, score in retrieved_docs:

        context += doc.page_content + "\n\n"

        retrieved_chunks.append(
            {
                "page": doc.metadata.get("page", "Unknown"),
                "score": float(score)
            }
        )

    return context, retrieved_chunks



@app.post("/groq_chat")
def groq_chat(data: UserQuery):

    context, retrieved_chunks = build_context(data.query)

    prompt = f"""
You are a question answering assistant.

Answer ONLY from the context.

If the answer is not present, reply exactly:

I don't know from the document.

Context:
{context}

Question:
{data.query}

Answer:
"""

    response = groq_llm.invoke(prompt)

    return {
        "query": data.query,
        "answer": response.content,
        "retrieved_chunks": retrieved_chunks
    }



@app.post("/opensource_chat")
def opensource_chat(data: UserQuery):

    context, retrieved_chunks = build_context(data.query)

    prompt = f"""
You are a question answering assistant.

Answer ONLY from the context.

If the answer is not present, reply exactly:

I don't know from the document.

Context:
{context}

Question:
{data.query}

Answer:
"""

    result = opensource_llm(prompt)

    answer = result[0]["generated_text"]

    return {
        "query": data.query,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks
    }


@app.get("/")
def home():

    return {
        "message": "RAG API Running Successfully"
    }