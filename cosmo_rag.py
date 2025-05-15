
import os
import json
import uuid
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, PartitionKey
from openai import AzureOpenAI


load_dotenv()

COSMOS_DB_CONNECTION_STRING = os.getenv("COSMOS_DB_CONNECTION_STRING")
DATABASE_NAME = os.getenv("DATABASE_NAME")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
GPT_ENGINE = os.getenv("GPT_ENGINE")
EMBEDDING_ENGINE = os.getenv("EMBEDDING_ENGINE")

#  CosmosDB Client 
client = CosmosClient.from_connection_string(COSMOS_DB_CONNECTION_STRING)
database = client.create_database_if_not_exists(id=DATABASE_NAME)

#  Vector Embedding & Index Policy
pk = "/category"
vector_embedding_policy = {
    "vectorEmbeddings": [{
        "path": "/vector",
        "dataType": "float32",
        "distanceFunction": "cosine",
        "dimensions": 1536
    }]
}

indexing_policy = {
    "vectorIndexes": [{
        "path": "/vector",
        "type": "diskANN"
    }]
}

# Create Container with Vector Support 
try:
    container = database.create_container_if_not_exists(
        id=CONTAINER_NAME,
        partition_key=PartitionKey(path=pk),
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_embedding_policy
    )
except Exception as e:
    print("Error creating container:", e)

#  Azure OpenAI Client 
azure_openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Embedding Generator Function 
def generate_embeddings(client, text):
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_ENGINE
    )
    return response.model_dump()['data'][0]['embedding']

# Load and Process Food Items Dataset 
file_path = "./food_items.json"

with open(file_path) as f:
    data = json.load(f)

for obj in data:
    obj['id'] = str(uuid.uuid4())
    obj['vector'] = generate_embeddings(azure_openai_client, obj['description'])
    container.upsert_item(obj)

with open("./new_dataset.json", 'w') as f:
    json.dump(data, f)

# === User Query Embedding & Vector Search ===
user_query = "are pizzas available? i am lactose intolerant"
user_embeddings = generate_embeddings(azure_openai_client, user_query)

query_text = f"""
SELECT TOP 5 c.category, c.name, c.description, c.price,
VectorDistance(c.vector, {user_embeddings}) AS SimilarityScore
FROM c
ORDER BY VectorDistance(c.vector, {user_embeddings})
"""

results = container.query_items(
    query=query_text,
    enable_cross_partition_query=True
)

dishes = [item for item in results]

# GPT Engine Summarization 
system_message = """
You are meant to behave as a RAG chatbot that derives its context from a database of food items stored in Azure CosmosDB for NoSQL API.
Please answer strictly from the context from the database provided and if you don't have an answer, politely say so.
Don't include any extra information or links.
The context will be in the form of a list of food items with the structure:
"category", "name", "description", "price".
Provide professional answers without revealing the use of RAG.
"""

user_message = f"""
User query: {user_query}
Context: {dishes}
"""

chat_response = azure_openai_client.chat.completions.create(
    model=GPT_ENGINE,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ],
    temperature=0.7
)

print("\n GPT Response \n")
print(chat_response.choices[0].message.content)
