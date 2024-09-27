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
    res = chain_extract.invoke, input={'page_data': page_data}
    
    # Parse the result
    json_parser = JsonOutputParser()
    json_res = json_parser.parse(res.content)
    
    # Extract skills from the JSON output
    job = json_res
    if not json_res or not isinstance(json_res, list):
        raise ValueError("Invalid JSON format or no data found.")
    skills = json_res[0].get('skills', [])
    return skills



# FastAPI route to query the collection using a POST request
@app.post("/search")
async def search(request: SearchRequest):
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
        
        chain_extract = prompt_extract | llm
        res = chain_extract.invoke(input={'page_data': request.query_text})

        # Log the raw output for debugging
        print("Raw output from model:", res.content)

        json_parser = JsonOutputParser()
        json_res = json_parser.parse(res.content)

        # Check if json_res is a valid list or dict
        if isinstance(json_res, dict) and 'skills' in json_res:
            # Handle the case where json_res is a dict with skills
            all_skills = json_res.get('skills', [])
        elif isinstance(json_res, list):
            # Handle the case where json_res is a list of jobs
            all_skills = []
            for job in json_res:
                if isinstance(job, dict) and 'skills' in job:
                    all_skills.extend(job['skills'])
        else:
            raise ValueError("Expected a list or dict but got: {}".format(type(json_res)))

        # Make the query using the list of skills
        links = collection.query(query_texts=all_skills, n_results=2).get('metadatas', [])

        return links
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
