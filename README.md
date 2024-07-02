# Rag Services

## System Architecture
![Model](img/architecture.png)

## Components
### Compute Service:
This is the heavy computation services of the system.
- **LLM Service** - LLM is up and running here
- **Embedding Service** - Sentence/Document embedding service is running here
- **Translator Service** - All direction translation service is running here

### Bot Backend:
This is the full RAG pipeline which answers a user query using the available knowledge bases fed to the system.

- **Bot Service** - The RAG pipeline
- **DB Service** - The RAG knowledge base store
### Client App:
Client frontend App that the user interacts with the bot/system.
