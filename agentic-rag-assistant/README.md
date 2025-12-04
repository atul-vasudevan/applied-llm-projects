# Agentic RAG Notes Assistant

A small, local agentic **RAG** (Retrieval-Augmented Generation) project that answers questions using a public open dataset instead of hand-written notes.

On first run, the dataset (AG News via HuggingFace datasets) is downloaded and converted into vector embeddings stored in a FAISS index for fast semantic search.
All reasoning and generation is done locally using Ollama.
The assistant uses:

- **LangGraph** to orchestrate the workflow
- **LangChain + FAISS + OllamaEmbeddings** for retrieval-augmented generation
- A local **Ollama** LLM for all reasoning and answers

The graph includes distinct decision nodes so the system can:

- Route the query to the right behavior:

    - **rag** → retrieve context from dataset

    - **chitchat** → answer freely without retrieval

    - **off_topic** → gracefully decline

- Inspect retrieval quality and avoid hallucination if context is weak

- Provide inline citations back to retrieved sources
