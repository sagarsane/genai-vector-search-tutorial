import os
import pymongo
from openai import OpenAI

import requests
from pprint import pprint

# Retrieve MongoDB connection details from environment variables
mongo_uri = os.environ.get("MONGO_URI")
mongo_db = os.environ.get("MONGO_DB")
mongo_collection = os.environ.get("MONGO_COLLECTION")

# Create a MongoDB client
client = pymongo.MongoClient(mongo_uri)
db = client[mongo_db]
collection = db[mongo_collection]

# Retrieve OpenAI API token from environment variable
openAIApiKey = os.environ.get("OPENAI_API_KEY")
openAI_client = OpenAI(api_key=openAIApiKey)
def generate_embedding(text: str) -> list[float]:
    """
    Generates an embedding for a given text using the OpenAI API
    """    
    response = openAI_client.embeddings.create(model="text-embedding-ada-002",
    input=text)
    return response.data[0].embedding


query = "imaginary action characters from outer space at war."

results = collection.aggregate([
    {
        '$vectorSearch': {
            "queryVector": generate_embedding(query),
            "path" : "plot_embedding",
            "numCandidates": 100,
            "limit": 4,
            "index": "PlotSemanticSearch",
        }
    }
])

for document in results:
    print(f"Movie Name: {document['title']},\n Movie Plot: {document['plot']}\n")
