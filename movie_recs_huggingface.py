import os
import pymongo
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

# Retrieve HuggingFace API token from environment variable
hf_token = os.environ.get("HF_TOKEN")

embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embedding(text: str) -> list[float]:
    """
    Generates an embedding for a given text using the HuggingFace API
    """    
    headers = {"Authorization": f"Bearer {hf_token}"}
    data = {"inputs": text}
    response = requests.post(embedding_url, headers=headers, json=data)

    if response.status_code != 200:
        raise ValueError(f"Request failed with status code: {response.status_code}: {response.text}")
    return response.json()

# for doc in collection.find({'plot': {'$exists': True}}).limit(500):
#     doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
#     collection.replace_one({'_id': doc['_id']}, doc)

query = "comedy characters from outer space at war. Not sensual."

results = collection.aggregate([
    {
        '$vectorSearch': {
            "queryVector": generate_embedding(query),
            "path" : "plot_embedding_hf",
            "numCandidates": 100,
            "limit": 4,
            "index": "PlotSemanticSearch",
        }
    }
])

for document in results:
    print(f"Movie Name: {document['title']},\n Movie Plot: {document['plot']}\n")
