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
        Your job is to extract the relevant skills mentioned in the job descriptions and return them in JSON format.
        Each object in the JSON should have two keys: `role` and `skills`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):
        """
        )
        page_data = request.query_text
        chain_extract = prompt_extract | llm
        res = chain_extract.invoke(input={'page_data':page_data})
        json_parser = JsonOutputParser()
        json_res = json_parser.parse(res.content)
        job = json_res
        skills = job[0].get('skills', [])
        links = collection.query(query_texts=skills, n_results=2).get('metadatas', [])
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
