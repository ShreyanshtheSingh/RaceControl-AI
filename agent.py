"""
RaceControl AI — Intelligent F1 Regulations Assistant
======================================================
Capstone Project | Agentic AI Hands-On Course 2026
Domain  : 2026 FIA Formula 1 Sporting Regulations
User    : F1 fans, engineers, and students querying F1 rules
Success : Answers 90%+ of regulation questions faithfully;
          uses live F1 API for current standings/results;
          falls back to web search for general F1 news.

Architecture (LangGraph):
  memory → router → [retrieve | skip | tool] → answer → eval → save → END

Tool    : Jolpica F1 API (structured race data) + DuckDuckGo (fallback)
Deployment: Streamlit (capstone_streamlit.py)
"""

import os
import json
import re
import requests
from typing import TypedDict, List

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import chromadb
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2
EMBED_MODEL            = "all-MiniLM-L6-v2"
LLM_MODEL              = "llama-3.3-70b-versatile"
KB_FILE                = os.path.join(os.path.dirname(__file__), "data.json")
COLLECTION_NAME        = "f1_regulations_kb"

# Jolpica (Ergast) F1 API base — free, no key needed
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

# Keywords that suggest a live-data query (route → tool)
LIVE_KEYWORDS = [
    "current", "standings", "championship", "latest", "last race",
    "last grand prix", "who won", "winner", "results", "2025", "2026",
    "this season", "points table", "leaderboard", "schedule", "next race",
    "calendar", "tonight", "today", "recent", "news",
]


# ──────────────────────────────────────────────────────────────────────────────
# 1. LLM
# ──────────────────────────────────────────────────────────────────────────────
from langchain_core.language_models import BaseChatModel
def build_llm() -> BaseChatModel:
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        raise EnvironmentError("GROQ_API_KEY not found — check your .env file")
    
    from langchain_groq import ChatGroq
    groq_llm = ChatGroq(model=LLM_MODEL, temperature=0, max_retries=1)
    
    # Fallback to Gemini if Groq runs into Rate Limit
    from langchain_google_genai import ChatGoogleGenerativeAI
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    return groq_llm.with_fallbacks([gemini_llm])


# ──────────────────────────────────────────────────────────────────────────────
# 2. Knowledge Base — load data.json → ChromaDB
# ──────────────────────────────────────────────────────────────────────────────
def build_knowledge_base(embedder: SentenceTransformer):
    """Load 83 F1 regulation documents from data.json into ChromaDB."""
    with open(KB_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    client = chromadb.Client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    texts      = [d["text"]  for d in documents]
    ids        = [d["id"]    for d in documents]
    metadatas  = [{"topic": d["topic"]} for d in documents]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"✅ Knowledge base ready — {collection.count()} documents loaded")
    return collection, documents


# ──────────────────────────────────────────────────────────────────────────────
# 3. State
# ──────────────────────────────────────────────────────────────────────────────
class RaceControlState(TypedDict):
    # Input
    question:       str           # user's current question

    # Memory
    messages:       List[dict]    # full conversation history

    # Routing
    route:          str           # "retrieve" | "memory_only" | "tool"

    # RAG
    retrieved:      str           # formatted ChromaDB context chunks
    sources:        List[str]     # source topic names

    # Tool
    tool_result:    str           # output from Jolpica API or DuckDuckGo

    # Answer
    answer:         str           # final LLM response

    # Eval / quality control
    faithfulness:   float         # self-reflection score 0.0 – 1.0
    eval_retries:   int           # retry counter (max = MAX_EVAL_RETRIES)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Node functions
# ──────────────────────────────────────────────────────────────────────────────

def build_memory_node():
    """Append question to history; apply sliding window (last 6 messages)."""
    def memory_node(state: RaceControlState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:
            msgs = msgs[-6:]
        return {"messages": msgs}
    return memory_node


def build_router_node(llm: ChatGroq):
    """Route to retrieve / memory_only / tool based on the question."""
    def router_node(state: RaceControlState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])
        recent   = "; ".join(
            f"{m['role']}: {m['content'][:60]}"
            for m in messages[-3:-1]
        ) or "none"

        prompt = f"""You are a router for RaceControl AI, an F1 Sporting Regulations chatbot.

Available routes:
- retrieve    : search the knowledge base for questions about F1 rules, regulations, penalties,
                tyre allocations, session formats, pit lane rules, parc fermé, safety car, etc.
- memory_only : answer from conversation history only (e.g. "what did you just say?", "repeat that")
- tool        : use the live F1 data tool for current championship standings, recent race results,
                driver/constructor points, this season's calendar, or any question about live events

Recent conversation: {recent}
Current question    : {question}

Reply with ONLY one word: retrieve / memory_only / tool"""

        response = llm.invoke(prompt)
        decision = response.content.strip().lower().split()[0]
        if "memory" in decision:
            decision = "memory_only"
        elif "tool" in decision:
            decision = "tool"
        else:
            decision = "retrieve"
        return {"route": decision}
    return router_node


def build_retrieval_node(embedder: SentenceTransformer, collection):
    """Embed question, query ChromaDB top-3, return formatted context."""
    def retrieval_node(state: RaceControlState) -> dict:
        q_emb   = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks  = results["documents"][0]
        topics  = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(
            f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
        )
        return {"retrieved": context, "sources": topics}
    return retrieval_node


def skip_retrieval_node(state: RaceControlState) -> dict:
    return {"retrieved": "", "sources": []}


def build_tool_node():
    """
    Dual tool:
      1. Jolpica F1 API — structured data for standings / results / calendar
      2. DuckDuckGo — fallback for general live F1 news
    """
    def _jolpica_standings() -> str:
        try:
            # Driver standings
            r = requests.get(f"{JOLPICA_BASE}/current/driverStandings.json", timeout=6)
            data = r.json()
            standings = data["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
            lines = ["🏆 2025 F1 Drivers Championship Standings:"]
            for s in standings[:10]:
                drv  = s["Driver"]
                team = s["Constructors"][0]["name"] if s["Constructors"] else "—"
                lines.append(
                    f"  P{s['position']:>2}. {drv['givenName']} {drv['familyName']} "
                    f"({team}) — {s['points']} pts"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"[Standings API error: {e}]"

    def _jolpica_last_race() -> str:
        try:
            r = requests.get(f"{JOLPICA_BASE}/current/last/results.json", timeout=6)
            data = r.json()
            race  = data["MRData"]["RaceTable"]["Races"][0]
            name  = race["raceName"]
            date  = race["date"]
            lines = [f"🏁 Last Race: {name} ({date})"]
            for res in race["Results"][:5]:
                drv = res["Driver"]
                lines.append(
                    f"  P{res['position']:>2}. {drv['givenName']} {drv['familyName']} "
                    f"({res['Constructor']['name']})"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"[Race results API error: {e}]"

    def _jolpica_constructor_standings() -> str:
        try:
            r = requests.get(f"{JOLPICA_BASE}/current/constructorStandings.json", timeout=6)
            data = r.json()
            standings = data["MRData"]["StandingsTable"]["StandingsLists"][0]["ConstructorStandings"]
            lines = ["🏗️ 2025 F1 Constructors Championship Standings:"]
            for s in standings[:10]:
                lines.append(
                    f"  P{s['position']:>2}. {s['Constructor']['name']} — {s['points']} pts"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"[Constructor standings API error: {e}]"

    def _duckduckgo_search(question: str) -> str:
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(question + " Formula 1 F1", max_results=4))
            if not results:
                return "No web results found."
            return "\n".join(
                f"• {r['title']}: {r['body'][:280]}" for r in results
            )
        except Exception as e:
            return f"[Web search error: {e}]"

    def tool_node(state: RaceControlState) -> dict:
        question = state["question"].lower()

        # Route to best API endpoint
        if any(kw in question for kw in ["driver standing", "drivers standing", "drivers championship", "who leads"]):
            result = _jolpica_standings()
        elif any(kw in question for kw in ["last race", "last grand prix", "race result", "who won"]):
            result = _jolpica_last_race()
        elif any(kw in question for kw in ["constructor standing", "team standing", "constructor championship"]):
            result = _jolpica_constructor_standings()
        elif any(kw in question for kw in LIVE_KEYWORDS):
            # Try last race first, then fall back to DuckDuckGo
            result = _jolpica_last_race()
            if "error" in result.lower():
                result = _duckduckgo_search(state["question"])
        else:
            result = _duckduckgo_search(state["question"])

        return {"tool_result": result}
    return tool_node


def build_answer_node(llm: ChatGroq):
    """Build the grounded answer from KB context + tool result + history."""
    def answer_node(state: RaceControlState) -> dict:
        question     = state["question"]
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        messages     = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)

        context_parts = []
        if retrieved:
            context_parts.append(f"KNOWLEDGE BASE (2026 FIA F1 Regulations):\n{retrieved}")
        if tool_result:
            context_parts.append(f"LIVE F1 DATA:\n{tool_result}")
        context = "\n\n".join(context_parts)

        if context:
            system_content = f"""You are RaceControl AI — an expert F1 Sporting Regulations assistant \
powered by the 2026 FIA Formula One Sporting Regulations.

Answer using ONLY the information provided in the context below.
If the answer is not in the context, say clearly:
"I don't have that information in my knowledge base. For authoritative information, \
please check the FIA website at fia.com or the official F1 website at formula1.com."

Do NOT add information from your training data beyond what is in the context.
Be precise, cite tyre specs, times, and numbers exactly as they appear in the regulations.

{context}"""
        else:
            system_content = (
                "You are RaceControl AI — an expert F1 Sporting Regulations assistant. "
                "Answer based on the conversation history. If you don't know, say so."
            )

        if eval_retries > 0:
            system_content += (
                "\n\nIMPORTANT: Your previous answer did not meet quality standards. "
                "Answer using ONLY information explicitly stated in the provided context above."
            )

        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            if msg["role"] == "user":
                lc_msgs.append(HumanMessage(content=msg["content"]))
            else:
                lc_msgs.append(AIMessage(content=msg["content"]))
        lc_msgs.append(HumanMessage(content=question))

        response = llm.invoke(lc_msgs)
        return {"answer": response.content}
    return answer_node


def build_eval_node(llm: ChatGroq):
    """Score faithfulness; trigger retry if score < FAITHFULNESS_THRESHOLD."""
    def eval_node(state: RaceControlState) -> dict:
        answer  = state.get("answer", "")
        context = state.get("retrieved", "")[:500]
        retries = state.get("eval_retries", 0)

        if not context:
            # No retrieval (tool or memory-only) — skip faithfulness check
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
1.0 = fully faithful to context. 0.5 = some unsupported claims. 0.0 = mostly hallucinated.

Context: {context}
Answer : {answer[:300]}"""

        result = llm.invoke(prompt).content.strip()
        try:
            score = float(re.findall(r"\d+\.?\d*", result)[0])
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5

        gate = "✅ PASS" if score >= FAITHFULNESS_THRESHOLD else "⚠️  RETRY"
        print(f"  [eval] Faithfulness: {score:.2f} {gate}")
        return {"faithfulness": score, "eval_retries": retries + 1}
    return eval_node


def save_node(state: RaceControlState) -> dict:
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "assistant", "content": state["answer"]}]
    return {"messages": msgs}


# ──────────────────────────────────────────────────────────────────────────────
# 5. Routing decisions
# ──────────────────────────────────────────────────────────────────────────────

def route_decision(state: RaceControlState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool":        return "tool"
    if route == "memory_only": return "skip"
    return "retrieve"


def eval_decision(state: RaceControlState) -> str:
    score   = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"  # retry


# ──────────────────────────────────────────────────────────────────────────────
# 6. Graph assembly
# ──────────────────────────────────────────────────────────────────────────────

def build_graph(llm, embedder, collection):
    """Assemble and compile the full LangGraph StateGraph."""

    memory_node    = build_memory_node()
    router_node    = build_router_node(llm)
    retrieval_node = build_retrieval_node(embedder, collection)
    tool_node_fn   = build_tool_node()
    answer_node    = build_answer_node(llm)
    eval_node      = build_eval_node(llm)

    graph = StateGraph(RaceControlState)

    graph.add_node("memory",    memory_node)
    graph.add_node("router",    router_node)
    graph.add_node("retrieve",  retrieval_node)
    graph.add_node("skip",      skip_retrieval_node)
    graph.add_node("tool",      tool_node_fn)
    graph.add_node("answer",    answer_node)
    graph.add_node("eval",      eval_node)
    graph.add_node("save",      save_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")

    graph.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
    )

    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("answer",   "eval")

    graph.add_conditional_edges(
        "eval", eval_decision,
        {"answer": "answer", "save": "save"}
    )
    graph.add_edge("save", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    print("✅ RaceControl AI graph compiled successfully!")
    return app


# ──────────────────────────────────────────────────────────────────────────────
# 7. Public ask() helper
# ──────────────────────────────────────────────────────────────────────────────

def ask(app, question: str, thread_id: str = "default") -> dict:
    """Invoke the agent and return the full state dict."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"question": question}, config=config)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 8. Initialise everything (called once at startup)
# ──────────────────────────────────────────────────────────────────────────────

def initialise():
    """Load models, KB, and compile graph. Returns (app, embedder, collection, documents)."""
    print("Loading LLM...")
    llm = build_llm()

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Building knowledge base from data.json...")
    collection, documents = build_knowledge_base(embedder)

    print("Assembling LangGraph...")
    app = build_graph(llm, embedder, collection)

    return app, embedder, collection, documents


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test when run as a script
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app, embedder, collection, documents = initialise()

    test_q = "What is the pit lane speed limit during a competition?"
    print(f"\nTest query: {test_q}")
    result = ask(app, test_q, thread_id="smoke-test")
    print(f"Answer: {result['answer']}")
    print(f"Route : {result.get('route', '?')} | Faithfulness: {result.get('faithfulness', 0):.2f}")
