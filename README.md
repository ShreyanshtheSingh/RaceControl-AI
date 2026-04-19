# 🏁 RaceControl AI

**Agentic F1 Sporting Regulations Assistant**

> An intelligent, multi-agent conversational assistant that answers questions about the 2026 FIA Formula One Sporting Regulations — and fetches live championship standings in real time.

---

## 📌 Overview

RaceControl AI is a capstone project for the **Agentic AI (IS30056)** course. It bridges a domain-specific knowledge base (the 2026 FIA F1 Sporting Regulations) with live race data, using a graph-based reasoning architecture to deliver hallucination-free, faithfully grounded answers.

Built for F1 fans, race engineers, motorsports students, and aspiring stewards who need fast, accurate answers without manually parsing hundreds of pages of regulation PDFs.

---

## ✨ Features

- **Graph-Based Reasoning** — 8-node finite state machine built with LangGraph, structuring every query into distinct, observable decision steps: `memory → router → [retrieve | skip | tool] → answer → eval → save → END`
- **Domain-Specific RAG** — ChromaDB vector store loaded with 83 regulation document chunks, embedded via `all-MiniLM-L6-v2` and retrieved with semantic similarity search
- **Agentic Dual-Tool Routing** — Live F1 data via the [Jolpica F1 API](https://api.jolpi.ca/ergast/f1) (standings, results, calendar), with DuckDuckGo as a web-search fallback
- **Self-Correcting Faithfulness Evaluation** — LLM scores its own answers against retrieved context; if the score falls below 0.7, the answer is automatically regenerated before being shown to the user
- **LLM Fallback Chaining** — Primary LLM is Llama 3.3 70B via Groq; silently falls back to Gemini 1.5 Flash on rate limit errors, keeping sessions uninterrupted
- **Conversational Memory** — Sliding window of the last 6 messages maintained per session thread
- **F1-Themed Streamlit UI** — Dark-mode interface inspired by F1 telemetry screens, with collapsible debug panels for route, context sources, and faithfulness scores

---

## 🏗️ Architecture

```
memory → router → ┌─ retrieve  (ChromaDB semantic search)
                  ├─ skip      (memory-only, no retrieval)
                  └─ tool      (Jolpica API / DuckDuckGo)
                         ↓
                       answer
                         ↓
                        eval  ──(retry if faithfulness < 0.7)──┐
                         ↓                                      │
                        save ←──────────────────────────────────┘
                         ↓
                        END
```

The **router node** classifies each incoming question into one of three routes:

| Route | When Used |
|---|---|
| `retrieve` | F1 rules, penalties, tyre specs, pit lane, safety car, parc fermé, etc. |
| `tool` | Live standings, race results, driver points, season calendar |
| `memory_only` | Follow-up questions referring to the current conversation |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph + LangChain |
| Primary LLM | Llama 3.3 70B (`llama-3.3-70b-versatile`) via Groq |
| Fallback LLM | Gemini 1.5 Flash via Google GenerativeAI |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers / HuggingFace) |
| Vector DB | ChromaDB (in-memory) |
| Live Data | Jolpica F1 REST API + DuckDuckGo |
| Frontend | Streamlit |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision) |

---

## 📁 Project Structure

```
racecontrol-ai/
├── agent.py                  # Core LangGraph agent (nodes, graph, ask())
├── capstone_streamlit.py     # Streamlit web interface
├── data.json                 # 83 F1 regulation document chunks (knowledge base)
├── .env                      # API keys (not committed)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Soumil0707/RaceControl-AI.git
cd RaceControl-AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include: `langchain`, `langgraph`, `langchain-groq`, `langchain-google-genai`, `chromadb`, `sentence-transformers`, `streamlit`, `requests`, `python-dotenv`

### 3. Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here   # optional — used for fallback LLM
```

> The Jolpica F1 API requires no key.

### 4. Run the Streamlit app

```bash
streamlit run capstone_streamlit.py
```

The first load takes ~60 seconds as it builds the ChromaDB knowledge base and loads the embedding model.

### 5. Smoke-test the agent directly

```bash
python agent.py
```

---

## 💬 Example Queries

| Type | Example |
|---|---|
| 📖 Regulations | *"What is the pit lane speed limit during a competition?"* |
| 📖 Regulations | *"How many power unit elements are allowed per season?"* |
| 📖 Regulations | *"What happens in parc fermé?"* |
| 📖 Regulations | *"When is the Virtual Safety Car deployed?"* |
| 📖 Regulations | *"What must a driver do when serving a Drive-Through penalty?"* |
| 🌐 Live Data | *"Who leads the Drivers' Championship?"* |
| 🌐 Live Data | *"What were the results of the last Grand Prix?"* |
| 💬 Memory | *"Can you repeat that last point?"* |

---

## 📊 Evaluation Results

Benchmarked using a custom test suite and RAGAS:

- **10/11** test cases passed
- **RAGAS Faithfulness: 1.000** on static RAG queries
- **Faithfulness threshold:** 0.7 (answers below this are automatically regenerated, up to 2 retries)

---

## 🔮 Future Improvements

- **Hybrid Search** — Fuse BM25 sparse retrieval with ChromaDB dense vectors for better handling of exact Article numbers and operational acronyms (TTCS, LTCS, VSC)
- **FIA Timing WebSocket** — Real-time telemetry integration to support live session companion use
- **Expanded KB** — Add Technical and Financial Regulations alongside the Sporting Regulations

---

## 👤 Author

**Soumil Aggarwal** · Roll No. 2305016  
Course: Agentic AI (IS30056) · April 2026

---

## 📄 License

This project is submitted as a capstone for academic purposes. The 2026 FIA F1 Sporting Regulations remain the intellectual property of the Fédération Internationale de l'Automobile (FIA).
