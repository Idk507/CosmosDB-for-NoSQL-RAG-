# CosmosDB RAG Chatbot for Food Dataset

This project demonstrates Retrieval-Augmented Generation (RAG) using:

- **Azure CosmosDB for NoSQL with Vector Search (DiskANN)**
- **Azure OpenAI for Embeddings and Chat Completion**
- **Python SDKs and OpenAI APIs**

## Features

- Vectorize food item descriptions using OpenAI embeddings.
- Store items in CosmosDB with vector indexes.
- Perform semantic search using user queries.
- Use GPT to generate contextual answers based on top relevant items.

## Setup

```bash
pip install -r requirements.txt
