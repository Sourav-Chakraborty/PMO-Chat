from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from typing import List
from .extractPDF import extract_pdf_contents
from .prompts import documentSummeryPrompt,LLMQueryPrompt
import os
from  langchain_core.documents  import Document


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

promptForQuery = ChatPromptTemplate.from_template(LLMQueryPrompt)

promptForSummery = ChatPromptTemplate.from_template(documentSummeryPrompt)

class TextInput(BaseModel):
    texts: str

@app.post("/add-texts")
async def add_texts(
    files: List[UploadFile] = File(...),
    project_name: str = Form(...),
    project_date: str = Form(...),
    manager_name: str = Form(...),
    department: str = Form(None),
):
    extracted_results = []

    for pdf in files:
        # Simple validation for PDF
        if pdf.content_type != "application/pdf":
            continue  # or skip or return error
        pdf_bytes = await pdf.read()
        pdf_data = extract_pdf_contents(pdf_bytes)
        
        extracted_results.append({
            "filename": pdf.filename,
            "extracted": pdf_data
        })

    chain = promptForSummery | llm |  JsonOutputParser()
    
    answer = chain.invoke({
        "context": extracted_results
    })

    documents = []
    for item in answer:
        documents.append(Document(page_content=item["summary"], metadata={"title": item["title"], "category": item["category"]}))
    
    vectorstore.add_documents(documents)
    
    return {
       "response":"Embedding completed successfully",
       "documents":documents
    }

class QueryInput(BaseModel):
    query: str

@app.post("/chat")
def chat(body: QueryInput):
    results = vectorstore.similarity_search(body.query, k=3)

    context = "\n\n".join([doc.page_content for doc in results])

    chain = promptForQuery | llm | StrOutputParser()

    answer = chain.invoke({"context": context, "question": body.query})

    return {
        "question": body.query,
        "answer": answer,
        "sources": [doc.metadata for doc in results]
    }
