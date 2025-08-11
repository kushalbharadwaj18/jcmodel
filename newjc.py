# -*- coding: utf-8 -*-
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()

# Globals for lazy loading
qa_chain = None
embedding_model = None

# Build FAISS index only if not already saved
if not os.path.exists("faiss_index"):
    print("Building FAISS index...")
    file_path = "./data.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    print("File loaded successfully!")
    print(f"First 500 characters:\n{text_data[:500]}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text_data)
    print(f"Total chunks created: {len(chunks)}")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding_model)
    vector_store.save_local("faiss_index")
    print("FAISS vector store created and saved locally!")


def get_qa_chain():
    """Load models only when needed."""
    global qa_chain, embedding_model
    if qa_chain is None:
        print("Loading models for the first time...")

        if embedding_model is None:
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        model_name = "microsoft/phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.2
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

    return qa_chain


class QueryRequest(BaseModel):
    query: str


@app.post("/ask")
def ask_question(request: QueryRequest):
    chain = get_qa_chain()
    result = chain({"query": request.query})
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:200] for doc in result["source_documents"]]
    }


@app.get("/")
def home():
    return {"message": "RAG Model API is running!"}
