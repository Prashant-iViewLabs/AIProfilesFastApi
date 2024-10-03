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
class SkillsRequest(BaseModel):
    skills: str  # List of skills in the request body

class CRUDRequest(BaseModel):
    techstack: str
    links: str

class UpdateRequest(BaseModel):
    id: str
    techstack: str
    links: str


# FastAPI route to query the collection using a POST request
@app.post("/search")
async def search(request: SearchRequest):
    # Query the ChromaDB collection based on the input paragraph
    try:
        prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        This is a job description.
        Your job is to extract the relevant skills, tools, technologies, softwares,etc... mentioned in the job descriptions and return them in JSON format.
        Each object in the JSON should have two keys: `role` and `skills`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):
        """
        )
        chain_extract = prompt_extract | llm
        res = chain_extract.invoke(input={'page_data':request.query_text})
        json_parser = JsonOutputParser()
        json_res = json_parser.parse(res.content)
        jobs = json_res
        # Initialize an empty list to store all skills
        all_skills = []

        # Extract skills from each job role
        for job in jobs:
            all_skills.extend(job['skills'])  # Combine skills from each job role

        # Make the query using the list of skills
        links = collection.query(query_texts=all_skills).get('metadatas', [])
        # job['skills']
        # Query using the list of skills
        # links = collection.query(query_texts=job['skills'], n_results=2)

        return links

        # res = await extractSkillsfromParagraph(request.query_text)# Only take the first 10 skills
        # print(res)
        # links = collection.query(query_texts=res, n_results=2).get('metadatas', [])
        # return {"links": links}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during link search: {str(e)}")



# FastAPI route to add new data to the collection (optional)
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


# FastAPI route to check if the API is live
@app.get("/status")
async def check_status():
    return {"message": "It is live now"}