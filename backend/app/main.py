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

os.environ["GOOGLE_API_KEY"] ='AIzaSyAnwP93cloaMU3CqSEVhb9M763rRO_q970'
app = FastAPI()

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

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
Answer the question based ONLY on the following context:

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
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

@app.post("/search")
def search(body: QueryInput):
    results = vectorstore.similarity_search(body.query, k=3)

    context = "\n\n".join([doc.page_content for doc in results])

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"context": context, "question": body.query})

    return {
        "question": body.query,
        "answer": answer,
        "sources": [doc.metadata for doc in results]
    }
