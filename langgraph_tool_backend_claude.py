from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
from datetime import datetime
import sqlite3
import time
import requests
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

search_tool=DuckDuckGoSearchRun(region='us-en')


#Tools
@tool
def calculator(first_num:float,second_num:float,operation:str)->dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol:str)->dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    try:
        r=requests.get(
             "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": os.getenv("ALPHAVANTAGE_API_KEY"),
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {'error':str(e)}


tools=[search_tool,get_stock_price,calculator]
llm_with_tools=llm.bind_tools(tools)






# 3. State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 4. Nodes
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    response = llm_with_tools.invoke(state['messages'])
    return {"messages": [response]}

tool_node=ToolNode(tools)


conn=sqlite3.connect(database='Chatbot.db', check_same_thread=False)
# Checkpointer
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
conn.commit()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    checkpoint_threads = {}
    for checkpoint in checkpointer.list(None):
        thread_id = str(checkpoint.config['configurable']['thread_id'])
        checkpoint_data = checkpoint.checkpoint
        checkpoint_timestamp = checkpoint_data.get('ts')
        try:
            updated_at = datetime.fromisoformat(checkpoint_timestamp).timestamp()
        except (TypeError, ValueError):
            updated_at = time.time()

        messages = checkpoint_data.get('channel_values', {}).get('messages', [])
        first_user_message = next(
            (message.content for message in messages if isinstance(message, HumanMessage)),
            "",
        )
        title = " ".join(first_user_message.split())
        if len(title) > 32:
            title = title[:29].rstrip() + "..."
        title = title or "New chat"

        previous = checkpoint_threads.get(thread_id)
        if previous is None or updated_at > previous[0]:
            checkpoint_threads[thread_id] = (updated_at, title)

    for thread_id, (updated_at, title) in checkpoint_threads.items():
        conn.execute(
            """
            INSERT OR IGNORE INTO chat_threads (thread_id, title, updated_at)
            VALUES (?, ?, ?)
            """,
            (thread_id, title, updated_at),
        )
    conn.commit()

    rows = conn.execute(
        "SELECT thread_id FROM chat_threads ORDER BY updated_at DESC, thread_id"
    )
    return [row[0] for row in rows]


def retrieve_thread_titles():
    rows = conn.execute("SELECT thread_id, title FROM chat_threads")
    return {thread_id: title for thread_id, title in rows}


def save_thread_title(thread_id: str, title: str):
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