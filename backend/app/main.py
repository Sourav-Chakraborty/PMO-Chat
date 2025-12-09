from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
app = FastAPI()

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

vector_size = 768 
collection_name = os.getenv("QDRANT_COLLECTION")

vectorstore = QdrantVectorStore(
    client=client,  
    collection_name=collection_name,
    embedding=embeddings    
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0.1
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

You will receive:
1. CONTEXT retrieved from a vector database (may be empty or irrelevant)
2. A USER QUESTION

Rules:
- If the CONTEXT is relevant, use it.
- If the CONTEXT is empty, irrelevant, or unhelpful, IGNORE it completely.
- You are allowed to answer using your general world knowledge.
- Do NOT say "Based on the context" if you are not using it.

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
""")

class TextInput(BaseModel):
    texts: str

@app.post("/add-texts")
def add_texts(body: TextInput):
    print(body.texts)
    texts = [body.texts]
    vectorstore.add_texts(texts)
    return {"message": "Texts added to vectorstore"}

class QueryInput(BaseModel):
    query: str

@app.post("/chat")
def chat(body: QueryInput):
    results = vectorstore.similarity_search(body.query, k=3)

    context = "\n\n".join([doc.page_content for doc in results])

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"context": context, "question": body.query})

    return {
        "question": body.query,
        "answer": answer,
        "sources": [doc.metadata for doc in results]
    }
