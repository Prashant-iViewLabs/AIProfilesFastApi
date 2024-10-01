from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uuid
import chromadb
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatGroq(
    temperature=0,
    groq_api_key='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',
    model_name="llama-3.1-70b-versatile"
)
# Initialize FastAPI
app = FastAPI()

# Initialize ChromaDB persistent client
client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name="YPP-portfolio")

# Load the CSV into a DataFrame (assuming it only needs to be loaded once)
df = pd.read_csv("./Resources/profiles_techstack_-_resourcedata.csv")

# Populate the collection if it's empty
if not collection.count():
    for _, row in df.iterrows():
        collection.add(
            documents=[row["Techstack"]],
            metadatas={"links": row["Links"]},
            ids=[str(uuid.uuid4())]
        )

# Pydantic model for the input data
class SearchRequest(BaseModel):
    query_text: str
class SkillsRequest(BaseModel):
    skills: str  # List of skills in the request body

class CRUDRequest(BaseModel):
    techstack: str
    links: str

class UpdateRequest(BaseModel):
    id: str
    techstack: str
    links: str

async def extractSkillsfromParagraph(page_data: str):
    prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        This is a job description.
        Your job is to extract the relevant skills mentioned in the job descriptions and return them in JSON format.
        Each object in the JSON should have two keys: `role` and `skills`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):
        """
    )

    # Simulate async LLM invocation (if needed, adjust this for actual async API calls)
    chain_extract = prompt_extract | llm
    res = chain_extract.invoke(input={'page_data': page_data})
    
    # Parse the result
    json_parser = JsonOutputParser()
    json_res = json_parser.parse(res.content)
    
    # Extract skills from the JSON output
    job = json_res
    if not json_res or not isinstance(json_res, list):
        raise ValueError("Invalid JSON format or no data found.")
    skills = json_res[0].get('skills', [])
    return skills


# FastAPI route to extract skills using a POST request
@app.post("/extract_skills")
async def extract_skills(request: SearchRequest):
    try:
        skills = await extractSkillsfromParagraph(request.query_text)
        return {"skills": skills}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during skill extraction: {str(e)}")


# FastAPI route to query the collection using extracted skills
@app.post("/search_links")
async def search_links(request: SkillsRequest):
    try:
        # Make the query using the list of skills
        skillList = request.skills.split(',')
        links = collection.query(query_texts=skillList, n_results=2).get('metadatas', [])
        return links
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during link search: {str(e)}")


# FastAPI route to add new data to the collection
@app.post("/add")
async def add_document(crud_request: CRUDRequest):
    try:
        new_id = str(uuid.uuid4())
        collection.add(
            documents=[crud_request.techstack],
            metadatas={"links": crud_request.links},
            ids=[new_id]
        )
        return {"message": "Document added", "id": new_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during adding document: {str(e)}")


# FastAPI route to view a document in the collection
@app.get("/view/{id}")
async def view_document(id: str):
    try:
        document = collection.get(ids=[id])
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during document view: {str(e)}")


# FastAPI route to update a document in the collection
@app.put("/update")
async def update_document(update_request: UpdateRequest):
    try:
        collection.update(
            documents=[update_request.techstack],
            metadatas={"links": update_request.links},
            ids=[update_request.id]
        )
        return {"message": "Document updated", "id": update_request.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during document update: {str(e)}")


# FastAPI route to delete a document from the collection
@app.delete("/delete/{id}")
async def delete_document(id: str):
    try:
        collection.delete(ids=[id])
        return {"message": "Document deleted", "id": id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during document deletion: {str(e)}")

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import uuid
import chromadb
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatGroq(
    temperature=0,
    groq_api_key='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',
    model_name="llama-3.1-70b-versatile"
)
# Initialize FastAPI
app = FastAPI()

# Initialize ChromaDB persistent client
client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name="YPP-portfolio")

# Load the CSV into a DataFrame (assuming it only needs to be loaded once)
df = pd.read_csv("./Resources/profiles_techstack_-_resourcedata.csv")

# Populate the collection if it's empty
if not collection.count():
    for _, row in df.iterrows():
        collection.add(
            documents=[row["Techstack"]],
            metadatas={"links": row["Links"]},
            ids=[str(uuid.uuid4())]
        )

# Pydantic model for the input data
class SearchRequest(BaseModel):
    query_text: str

class CRUDRequest(BaseModel):
    techstack: str
    links: str

class UpdateRequest(BaseModel):
    id: str
    techstack: str
    links: str


async def extractSkillsfromParagraph(page_data: str):
    prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        This is a job description.
        Your job is to extract the relevant skills mentioned in the job descriptions and return them in JSON format.
        Each object in the JSON should have two keys: `role` and `skills`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):
        """
    )

    # Simulate async LLM invocation (if needed, adjust this for actual async API calls)
    chain_extract = prompt_extract | llm
    res = chain_extract.invoke(input={'page_data': page_data})
    
    # Parse the result
    json_parser = JsonOutputParser()
    json_res = json_parser.parse(res.content)
    
    # Extract skills from the JSON output
    job = json_res
    if not json_res or not isinstance(json_res, list):
        raise ValueError("Invalid JSON format or no data found.")
    skills = json_res[0].get('skills', [])
    return skills


# FastAPI route to extract skills using a POST request
@app.post("/extract_skills")
async def extract_skills(request: SearchRequest):
    try:
        skills = await extractSkillsfromParagraph(request.query_text)
        return {"skills": skills}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during skill extraction: {str(e)}")


# FastAPI route to query the collection using extracted skills
@app.post("/search_links")
async def search_links(skills: list):
    try:
        # Make the query using the list of skills
        links = collection.query(query_texts=skills, n_results=2).get('metadatas', [])
        return links
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during link search: {str(e)}")


# FastAPI route to add new data to the collection
@app.post("/add")
async def add_document(crud_request: CRUDRequest):
    try:
        new_id = str(uuid.uuid4())
        collection.add(
            documents=[crud_request.techstack],
            metadatas={"links": crud_request.links},
            ids=[new_id]
        )
        return {"message": "Document added", "id": new_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during adding document: {str(e)}")


# FastAPI route to view a document in the collection
@app.get("/view/{id}")
async def view_document(id: str):
    try:
        document = collection.get(ids=[id])
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during document view: {str(e)}")


# FastAPI route to update a document in the collection
@app.put("/update")
async def update_document(update_request: UpdateRequest):
    try:
        collection.update(
            documents=[update_request.techstack],
            metadatas={"links": update_request.links},
            ids=[update_request.id]
        )
        return {"message": "Document updated", "id": update_request.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during document update: {str(e)}")


# FastAPI route to delete a document from the collection
@app.delete("/delete/{id}")
async def delete_document(id: str):
    try:
        collection.delete(ids=[id])
        return {"message": "Document deleted", "id": id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during document deletion: {str(e)}")


# FastAPI route to view all documents with pagination
@app.get("/view_all")
async def view_all_documents(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1)):
    try:
        total_documents = collection.count()
        total_pages = (total_documents + page_size - 1) // page_size  # Calculate total number of pages

        if page > total_pages:
            raise HTTPException(status_code=400, detail=f"Page {page} is out of range. Total pages: {total_pages}")

        # Calculate the start and end indices for pagination
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_documents)

        # Retrieve the documents from the collection
        all_docs = collection.get(limit=page_size, offset=start_idx)

        # If no documents found
        if not all_docs:
            raise HTTPException(status_code=404, detail="No documents found")

        return {
            "page": page,
            "total_pages": total_pages,
            "total_documents": total_documents,
            "documents": all_docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during document retrieval: {str(e)}")
