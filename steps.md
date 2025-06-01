# GenAI Project Setup Checklist (Prompting + RAG + LangChain/LangGraph)

---

## Project Setup
- [.] Setup GitHub repository + initialize git
- [ ] Set up virtual environment (`venv` or `conda`)
- [ ] Create `requirements.txt` and folder structure
- [ ] Add `.gitignore`
- [ ] Create `.env` file for API keys (OpenAI, Pinecone, etc.)

---

## Knowledge Base Preparation
- [ ] Define input Files
- [ ] Write Indexing scripts 
    - [ ] Loader
    - [ ] Splitter
    - [ ] Choose Augmentation model (OpenAI, Gemini etc.)
    - [ ] Choose embedding model (OpenAI, HuggingFace, etc.)
- [ ] Choose vector DB (FAISS, Chroma, Pinecone, Weaviate)
- [ ] Augmentation
- [ ] Retriever

---

## Conditional Prompting Using RAG
- [ ] Design system prompts and templates
- [ ] Define Condition-Behaviour (CB) pairs
- [ ] Write Indexing scripts for CB pairs and embed the conditions into VDB
    - [ ] Loader
    - [ ] Condition Splitter 
    - [ ] Condition Augmentation
    - [ ] Condition Behaviour Indexing/mapping and embedding
    - [ ] Choose vector DB (FAISS, Chroma, Pinecone, Weaviate)
- [ ] Test and refine prompts

---

## RAG Pipeline
- [ ] Use Retriever to fetch instruction based on current condition 
- [ ] Use Retriever to fetch contextual knowledge
- [ ] Connect retriever → prompt → LLM chain


---

## User Upload System
- [ ] Allow users to upload instructions (text, JSON) and knowledge base (files, URLs)
- [ ] Dynamically ingest uploaded knowledge base
- [ ] Update vector store or retriever as needed

---

## Testing & Evaluation
- [ ] Create unit tests for:
    - Prompt behavior
    - Knowledge retrieval quality
    - Output consistency
- [ ] Run synthetic test cases or benchmarks

---

## UI or API Layer (optional)
- [ ] Build Streamlit, Gradio, or FastAPI interface
- [ ] Enable user interaction, upload, and query
- [ ] Deploy on HuggingFace Spaces, Render, etc.

---

## Logging & Monitoring
- [ ] Add logging (LangChain callbacks, `logging` module)
- [ ] Track retrieval hits, LLM responses, latency
- [ ] Set up feedback loop for continuous improvement

---
