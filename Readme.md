# Gen AI System

🚀 **AI system for user-defined instructions and knowledge-based responses**

---

## Problem Statement

Build an AI system where users can upload their **set of instructions** and **knowledge base**, and the AI will:

* Behave according to the given instructions
* Use the provided knowledge base as needed

---

## Features

✅ **Key-Value Extraction**
Extract structured information (entities, key-value pairs) from convertation history.

✅ **User Profile Building**
Construct dynamic user profiles from user's convertation via extracted data.

✅ **Context Awareness**
Maintain conversational and session context for coherent multi-turn interactions.

✅ **RAG for Memory-based Responses**
Retrieve past memory or knowledge chunks to enrich LLM outputs.

✅ **RAG for Conditional Behavioral Instructions**
Handle complex conditional logic using retrieval + generation workflows.

---

## Project Structure

```
/ai-system/
├── data/             # Raw, processed data, embeddings
├── models/           # Base models, fine-tuned models, RAG components
├── services/         # Core services (extraction, profiling, context, rag)
├── api/              # API layer (routes, schemas, middleware)
├── configs/          # Configuration files (models, RAG, extraction rules)
├── tests/            # Unit & integration tests
├── utils/            # Helpers, logger, preprocessors
├── notebooks/        # Experimentation and analysis
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container setup
└── README.md         # Project documentation
```

---

## Setup Instructions

1️⃣ **Clone the repository**

```bash
git clone <repo-url>
cd ai-system
```

2️⃣ **Create virtual environment & activate**

```bash
python -m venv AI_system
source AI_system/bin/activate  # on Windows: AI_system\Scripts\activate
```

3️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

4️⃣ **Set up environment variables**

* Create a `.env` file or update `config.yaml` with:

  * API keys (if using external LLMs)
  * Model paths
  * Database configs

5️⃣ **Run the API server**

```bash
uvicorn api.main:app --reload
```

---

## Tech Stack

* **LLMs** → OpenAI / Hugging Face transformers / custom fine-tuned models
* **Vector DB** →  Chroma for embeddings + retrieval
* **Backend** → FastAPI (for async APIs)
* **RAG Orchestration** → LangChain and LanGraph
* **Containerization** → Docker


---

## Future
* **Metadata Embedding Fusion** → Consider combining metadata (like category name or source) into embedding input to improve representation.
* **Add rechunking in chunks merge** → Consider rechunking the chunk classes if needed(large classes) after merging them to maintain the Granularity of the Information.



## Contributing

Pull requests are welcome! Please open issues or submit suggestions to improve this project.

