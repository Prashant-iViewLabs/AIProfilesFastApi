from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uuid
import chromadb

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


# FastAPI route to query the collection using a POST request
@app.post("/search")
async def search(request: SearchRequest):
    # Query the ChromaDB collection based on the input paragraph
    try:
        links = collection.query(query_texts=[request.query_text], n_results=2).get('metadatas', [])
        return {"links": links}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")


# FastAPI route to add new data to the collection (optional)
@app.post("/add")
async def add_document(techstack: str, links: str):
    new_id = str(uuid.uuid4())
    collection.add(
        documents=[techstack],
        metadatas={"links": links},
        ids=[new_id]
    )
    return {"message": "Document added", "id": new_id}
