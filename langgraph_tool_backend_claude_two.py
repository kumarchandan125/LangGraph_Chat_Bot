"""LangGraph chatbot backend: config, LLM, tools, graph, and chat-thread storage."""
import operator
import os
import sqlite3
import threading
import time
from datetime import datetime
from functools import wraps
from typing import Annotated, Literal, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ================================ Config ============================================

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DB_PATH = os.getenv("CHAT_DB_PATH", "Chatbot.db")

MAX_HISTORY_MESSAGES = 30   # messages sent to the LLM each turn (older ones stay in the DB)
SEARCH_MAX_CHARS = 2500     # cap web results so the LLM reads less and answers faster
SEARCH_CACHE_SECONDS = 300
STOCK_CACHE_SECONDS = 60
TITLE_LENGTH = 32

SYSTEM_PROMPT = """You are a fast, friendly assistant with access to tools.

Tools:
- web_search: current events, news, and facts that may have changed. Use short, specific queries.
- get_stock_price: live quotes for a ticker symbol such as AAPL or TSLA.
- calculator: any arithmetic. Never do maths in your head.
Call independent tools together in one step. Skip tools for things you already know well, and never invent tool results.

Style:
- Answer first, then add detail only if it helps. Keep replies concise.
- Use Markdown lightly: short paragraphs, bullets for lists, tables only for comparisons.
- Give numbers with units or currency and say when the data is from.
- If a tool fails or returns nothing, say so in one sentence and suggest what to try next.
- Ask a clarifying question only when the request is truly ambiguous.
- Reply in the user's language.

Today is {today}."""


# ================================ Helpers ===========================================

def ttl_cache(seconds: int, maxsize: int = 128):
    """Tiny thread-safe cache. Exceptions are never cached, so failures retry."""
    def decorator(fn):
        store: dict = {}
        lock = threading.Lock()

        @wraps(fn)
        def wrapper(*args):
            now = time.monotonic()
            with lock:
                hit = store.get(args)
                if hit and now - hit[0] < seconds:
                    return hit[1]
            value = fn(*args)
            with lock:
                if len(store) >= maxsize:
                    store.pop(next(iter(store)))
                store[args] = (now, value)
            return value
        return wrapper
    return decorator


def message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else b.get("text", "")
            for b in content
            if isinstance(b, str) or (isinstance(b, dict) and isinstance(b.get("text"), str))
        )
    return ""


def make_title(message: str, max_length: int = TITLE_LENGTH) -> str:
    title = " ".join(message.split())
    if len(title) > max_length:
        title = title[: max_length - 3].rstrip() + "..."
    return title or "New chat"


# ================================ Tools =============================================

_http = requests.Session()  # reuses connections between calls
_ddg = DuckDuckGoSearchRun(region="us-en")

_OPERATIONS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
}


@ttl_cache(SEARCH_CACHE_SECONDS)
def _run_search(query: str) -> str:
    return _ddg.invoke(query)[:SEARCH_MAX_CHARS]


@ttl_cache(STOCK_CACHE_SECONDS)
def _fetch_quote(symbol: str) -> dict:
    response = _http.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": os.getenv("ALPHAVANTAGE_API_KEY"),
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    quote = data.get("Global Quote") or {}
    if not quote:
        # Alpha Vantage returns HTTP 200 with a message on rate limits / bad symbols
        raise ValueError(data.get("Note") or data.get("Information") or f"No data for '{symbol}'")
    return {
        "symbol": quote.get("01. symbol"),
        "price": quote.get("05. price"),
        "change": quote.get("09. change"),
        "change_percent": quote.get("10. change percent"),
        "previous_close": quote.get("08. previous close"),
        "volume": quote.get("06. volume"),
        "latest_trading_day": quote.get("07. latest trading day"),
    }


@tool
def web_search(query: str) -> str:
    """Search the web for current events, news, and facts you don't know. Use short, specific queries."""
    try:
        return _run_search(query.strip())
    except Exception as exc:
        return f"Search failed: {exc}"


@tool
def get_stock_price(symbol: str) -> dict:
    """Get the latest stock price for a ticker symbol (e.g. 'AAPL', 'TSLA')."""
    try:
        return _fetch_quote(symbol.strip().upper())
    except (requests.RequestException, ValueError) as exc:
        return {"error": str(exc)}


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: Literal["add", "sub", "mul", "div"],
) -> dict:
    """Perform a basic arithmetic operation on two numbers (add, sub, mul, div)."""
    if operation == "div" and second_num == 0:
        return {"error": "Division by zero is not allowed"}
    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": _OPERATIONS[operation](first_num, second_num),
    }


TOOLS = [web_search, get_stock_price, calculator]


# ================================ LLM ===============================================

def build_llm() -> ChatGoogleGenerativeAI:
    kwargs = {"model": MODEL_NAME, "max_retries": 2, "timeout": 60}
    # Optional tuning via .env. Only passed when set, so defaults stay safe.
    if os.getenv("LLM_TEMPERATURE"):
        kwargs["temperature"] = float(os.environ["LLM_TEMPERATURE"])
    if os.getenv("LLM_THINKING_LEVEL"):
        kwargs["thinking_level"] = os.environ["LLM_THINKING_LEVEL"]
    return ChatGoogleGenerativeAI(**kwargs)


llm_with_tools = build_llm().bind_tools(TOOLS)


# ================================ Graph =============================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """Call the LLM with a fresh system prompt and a trimmed history."""
    history = trim_messages(
        state["messages"],
        max_tokens=MAX_HISTORY_MESSAGES,
        token_counter=len,          # count messages, not tokens (fast, no extra API call)
        strategy="last",
        start_on="human",           # never start mid tool-call
        include_system=False,
    )
    system = SystemMessage(SYSTEM_PROMPT.format(today=datetime.now().strftime("%A, %d %B %Y")))
    return {"messages": [llm_with_tools.invoke([system, *history])]}


# ================================ Storage ===========================================

_db_lock = threading.RLock()
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")      # faster writes, readers don't block
conn.execute("PRAGMA synchronous=NORMAL")
checkpointer = SqliteSaver(conn=conn)
checkpointer.setup()
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS chat_threads (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """
)
conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_threads_updated ON chat_threads(updated_at DESC)")
conn.commit()

_builder = StateGraph(ChatState)
_builder.add_node("chat_node", chat_node)
_builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
_builder.add_edge(START, "chat_node")
_builder.add_conditional_edges("chat_node", tools_condition)
_builder.add_edge("tools", "chat_node")

chatbot = _builder.compile(checkpointer=checkpointer)


def _backfill_thread(thread_id: str) -> None:
    """One-time import of a pre-existing checkpoint thread into chat_threads."""
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    first_user = next(
        (message_text(m.content) for m in messages if isinstance(m, HumanMessage)), ""
    )
    try:
        updated_at = datetime.fromisoformat(state.created_at).timestamp()
    except (TypeError, ValueError):
        updated_at = time.time()
    with _db_lock:
        conn.execute(
            "INSERT OR IGNORE INTO chat_threads (thread_id, title, updated_at) VALUES (?, ?, ?)",
            (thread_id, make_title(first_user), updated_at),
        )
        conn.commit()


def retrieve_all_threads() -> list:
    """Thread ids, newest first. Only unknown threads are loaded from checkpoints."""
    with _db_lock:
        known = {r[0] for r in conn.execute("SELECT thread_id FROM chat_threads")}
        in_checkpoints = [r[0] for r in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")]
    for thread_id in in_checkpoints:
        if thread_id not in known:
            _backfill_thread(thread_id)
    with _db_lock:
        rows = conn.execute("SELECT thread_id FROM chat_threads ORDER BY updated_at DESC, thread_id")
        return [row[0] for row in rows]


def retrieve_thread_titles() -> dict:
    with _db_lock:
        return dict(conn.execute("SELECT thread_id, title FROM chat_threads").fetchall())


def save_thread_title(thread_id: str, title: str) -> None:
    with _db_lock:
        conn.execute(
            """
            INSERT INTO chat_threads (thread_id, title, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                title = excluded.title,
                updated_at = excluded.updated_at
            """,
            (str(thread_id), title, time.time()),
        )
        conn.commit()