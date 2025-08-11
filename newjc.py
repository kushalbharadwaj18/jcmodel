# -*- coding: utf-8 -*-

from fastapi import FastAPI
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

# ===== 1. Load data and create vector store =====
file_path = "./data.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(text_data)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(chunks, embedding_model)
vector_store.save_local("faiss_index")

# ===== 2. Load FAISS retriever =====
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ===== 3. Load smaller model (fits in Render free tier) =====
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.2
)
llm = HuggingFacePipeline(pipeline=pipe)

# ===== 4. Create QA chain =====
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ===== 5. FastAPI App =====
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "RAG Model API is running!"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        result = qa_chain({"query": request.query})
        return {
            "answer": result["result"],
            "sources": [doc.page_content[:200] for doc in result["source_documents"]]
        }
    except Exception as e:
        return {"error": str(e)}
