from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Use small embeddings to save memory
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# Load FAISS index
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Load Mistral model (this is heavy for Render free tier)
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
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
    max_new_tokens=256,
    temperature=0.2
)
llm = HuggingFacePipeline(pipeline=pipe)

# Build Retrieval QA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "RAG Model API with Mistral is running!"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    result = qa_chain({"query": request.query})
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:200] for doc in result["source_documents"]]
    }
