"""Streamlit UI for the LangGraph tool-calling chatbot."""
from __future__ import annotations

import json
import time
import uuid
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from langgraph_tool_backend_claude import (
    chatbot,
    retrieve_all_threads,
    retrieve_thread_titles,
    save_thread_title,
)

st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬", initial_sidebar_state="expanded")

# **************************************** Config *************************************

STYLES_PATH = Path(__file__).with_name("styles.css")
PAINT_INTERVAL = 0.03        # seconds between UI repaints while streaming
MAX_SIDEBAR_THREADS = 30     # keep the sidebar light with many chats
MAX_TOOL_CHARS = 4000        # cap huge tool outputs rendered in the UI
TITLE_LENGTH = 32
SUGGESTIONS = (
    "What's the price of AAPL right now?",
    "Multiply 1284 by 37",
    "Search the latest AI news",
)

HistoryItem = dict[str, Any]


# **************************************** Helpers ************************************

def inject_css() -> None:
    try:
        css = STYLES_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return  # app still works, just unthemed
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def new_thread_id() -> str:
    return str(uuid.uuid4())


def make_title(message: str, max_length: int = TITLE_LENGTH) -> str:
    title = " ".join(message.split())
    if len(title) > max_length:
        title = title[: max_length - 3].rstrip() + "..."
    return title or "New chat"


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str)
            or (isinstance(block, dict) and isinstance(block.get("text"), str))
        )
    return ""


def render_tool_result(content: Any) -> None:
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            if len(content) > MAX_TOOL_CHARS:
                content = content[:MAX_TOOL_CHARS] + "\n... (truncated)"
            st.text(content)
            return
    if isinstance(content, (dict, list)):
        st.json(content, expanded=2)
    else:
        st.text(str(content))


def messages_to_history(messages: Iterable[BaseMessage]) -> list[HistoryItem]:
    """Convert LangChain messages into the lightweight UI history format."""
    history: list[HistoryItem] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": extract_text(msg.content)})
        elif isinstance(msg, ToolMessage):
            history.append({
                "role": "assistant",
                "tool_result": True,
                "tool_name": msg.name or "Tool",
                "content": msg.content,
            })
        elif text := extract_text(msg.content):
            history.append({"role": "assistant", "content": text})
    return history


def export_markdown(history: list[HistoryItem]) -> str:
    lines = ["# Chat export", ""]
    for m in history:
        if m.get("tool_result"):
            lines.append(f"> Used tool: `{m['tool_name']}`\n")
        else:
            speaker = "You" if m["role"] == "user" else "Assistant"
            lines.append(f"**{speaker}:** {m['content']}\n")
    return "\n".join(lines)


# **************************************** State **************************************

def init_state() -> None:
    """Initialise session state once. Factories avoid hitting the DB on every rerun."""
    ss = st.session_state
    defaults = {
        "message_history": list,
        "thread_id": new_thread_id,
        "chat_threads": lambda: [str(t) for t in retrieve_all_threads()],
        "thread_titles": retrieve_thread_titles,
    }
    for key, factory in defaults.items():
        if key not in ss:
            ss[key] = factory()


def reset_chat() -> None:
    st.session_state.thread_id = new_thread_id()
    st.session_state.message_history = []


def open_thread(thread_id: str) -> None:
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    st.session_state.thread_id = thread_id
    st.session_state.message_history = messages_to_history(state.values.get("messages", []))


def set_prompt(prompt: str) -> None:
    st.session_state.pending_prompt = prompt


def register_turn(thread_id: str, first_message: str) -> None:
    """Persist the title and move this chat to the top of the sidebar list."""
    ss = st.session_state
    ss.thread_titles.setdefault(thread_id, make_title(first_message))
    save_thread_title(thread_id, ss.thread_titles[thread_id])
    ss.chat_threads = [thread_id, *(t for t in ss.chat_threads if t != thread_id)]


# **************************************** UI pieces **********************************

def render_sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.title("💬 LangGraph Chatbot")
        st.button("＋  New chat", on_click=reset_chat, type="primary", key="new_chat")

        st.markdown("### Conversations")
        for tid in ss.chat_threads[:MAX_SIDEBAR_THREADS]:
            st.button(
                ss.thread_titles.get(tid, "New chat"),
                key=f"thread_{tid}",
                on_click=open_thread,
                args=(tid,),
                type="primary" if tid == ss.thread_id else "secondary",
            )

        if ss.message_history:
            st.markdown("### Export")
            st.download_button(
                "Download chat",
                data=export_markdown(ss.message_history),
                file_name="chat.md",
                mime="text/markdown",
                key="download_chat",
            )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>What can I look up for you?</h1>
            <p>I can search the web, check stock prices, and do the math.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for col, suggestion in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
        col.button(
            suggestion,
            key=f"suggest_{suggestion}",
            on_click=set_prompt,
            args=(suggestion,),
            use_container_width=True,
        )


def render_history(history: list[HistoryItem]) -> None:
    """Group consecutive messages by role so tools + text share ONE bubble."""
    for role, group in groupby(history, key=lambda m: m["role"]):
        with st.chat_message(role):
            for m in group:
                if m.get("tool_result"):
                    with st.status(f"✅ Used `{m['tool_name']}`", state="complete", expanded=False):
                        render_tool_result(m["content"])
                elif m["content"]:
                    st.markdown(m["content"])


# **************************************** Streaming **********************************

class TurnRenderer:
    """Turns LangGraph stream events into live UI and saves them to history."""

    def __init__(self, history: list[HistoryItem]) -> None:
        self.history = history
        self.text_box = st.empty()
        self.buffer = ""
        self.pending: dict[str, Any] = {}   # tool call key -> st.status widget
        self.seen: set[str] = set()
        self.last_paint = 0.0

    def handle(self, chunk: BaseMessage) -> None:
        if isinstance(chunk, ToolMessage):
            self._tool_finished(chunk)
        elif isinstance(chunk, AIMessage):
            calls = getattr(chunk, "tool_call_chunks", None) or getattr(chunk, "tool_calls", None) or []
            for call in calls:
                self._tool_started(call.get("id"), call.get("name"))
            self._append_text(extract_text(chunk.content))

    def close(self) -> None:
        self._flush_text()
        for entry in self.pending.values():
            entry.update(label="⚠️ Tool did not return", state="error")
        self.pending.clear()

    # -- internals --

    def _flush_text(self) -> None:
        if self.buffer.strip():
            self.text_box.markdown(self.buffer)  # final paint, no cursor
            self.history.append({"role": "assistant", "content": self.buffer})
        else:
            self.text_box.empty()
        self.buffer = ""

    def _append_text(self, text: str) -> None:
        if not text:
            return
        self.buffer += text
        now = time.monotonic()
        if now - self.last_paint >= PAINT_INTERVAL:
            self.text_box.markdown(self.buffer + "▌")
            self.last_paint = now

    def _tool_started(self, call_id: str | None, name: str | None) -> None:
        key = call_id or name
        if not name or key in self.seen:
            return
        self.seen.add(key)
        self._flush_text()
        self.pending[key] = st.status(f"🔧 Using `{name}`...", state="running", expanded=False)
        self.text_box = st.empty()  # following text renders BELOW the tool status

    def _tool_finished(self, msg: ToolMessage) -> None:
        name = msg.name or "Tool"
        entry = self.pending.pop(msg.tool_call_id, None) or self.pending.pop(name, None)
        if entry is None:
            entry = st.status(f"🔧 Using `{name}`...", state="running")
        entry.update(label=f"✅ Used `{name}`", state="complete", expanded=False)
        with entry:
            render_tool_result(msg.content)
        self.history.append({
            "role": "assistant",
            "tool_result": True,
            "tool_name": name,
            "content": msg.content,
        })


def stream_reply(user_input: str, thread_id: str) -> bool:
    """Stream one assistant turn. Returns True if it completed without error."""
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }
    with st.chat_message("assistant"):
        renderer = TurnRenderer(st.session_state.message_history)
        try:
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                renderer.handle(chunk)
        except Exception as exc:  # keep partial output, surface the error
            renderer.close()
            st.error(f"Something went wrong: {exc}")
            return False
        renderer.close()
    return True


# **************************************** App ****************************************

def main() -> None:
    inject_css()
    init_state()
    ss = st.session_state

    user_input = st.chat_input("Message the assistant...") or ss.pop("pending_prompt", None)

    if user_input:
        register_turn(ss.thread_id, user_input)
        ss.message_history.append({"role": "user", "content": user_input})

    render_sidebar()

    if not ss.message_history:
        render_hero()
    render_history(ss.message_history)

    if user_input and stream_reply(user_input, ss.thread_id):
        st.rerun()  # refresh sidebar title and export button


main()