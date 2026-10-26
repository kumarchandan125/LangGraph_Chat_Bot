import json
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_tool_backend import (
    chatbot,
    retrieve_all_threads,
    retrieve_thread_titles,
    save_thread_title,
)


# ========================= Utility Functions =========================

def generate_thread_id():
    return uuid.uuid4()


def add_thread(thread_id):
    """Add a thread without creating duplicates."""
    if not any(
        str(existing_id) == str(thread_id)
        for existing_id in st.session_state["chat_threads"]
    ):
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    """Start a new conversation."""
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id)


def load_conversation(thread_id):
    """Load messages saved by LangGraph."""
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    return state.values.get("messages", [])


def make_thread_title(message, max_length=32):
    """Generate a short title from the first user message."""
    title = " ".join(message.split())

    if len(title) > max_length:
        return title[: max_length - 3].rstrip() + "..."

    return title or "New chat"


def extract_text(content):
    """Extract readable text from common LangChain content formats."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict):
                text = block.get("text")

                if isinstance(text, str):
                    parts.append(text)

        return "".join(parts)

    return ""


def render_tool_result(content):
    """Render tool output as JSON when possible, otherwise as text."""
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            st.text(content)
            return

    if isinstance(content, (dict, list)):
        st.json(content, expanded=2)
    else:
        st.text(str(content))


def get_tool_name(message_chunk):
    """Extract a tool name from an AI tool-call chunk when available."""
    tool_calls = getattr(message_chunk, "tool_calls", None) or []

    for tool_call in tool_calls:
        if isinstance(tool_call, dict) and tool_call.get("name"):
            return tool_call["name"]

    tool_call_chunks = getattr(message_chunk, "tool_call_chunks", None) or []

    for tool_call in tool_call_chunks:
        if isinstance(tool_call, dict) and tool_call.get("name"):
            return tool_call["name"]

    return None


def reconstruct_message_history(messages):
    """Convert LangGraph messages into the UI's message-history format."""
    history = []

    for message in messages:
        if isinstance(message, HumanMessage):
            history.append({
                "role": "user",
                "content": extract_text(message.content),
            })

        elif isinstance(message, ToolMessage):
            history.append({
                "role": "assistant",
                "tool_result": True,
                "tool_name": message.name or "Tool",
                "content": message.content,
            })

        elif isinstance(message, AIMessage):
            content = extract_text(message.content)

            if content:
                history.append({
                    "role": "assistant",
                    "content": content,
                })

    return history


# ========================= Session Initialization =========================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_titles" not in st.session_state:
    titles = retrieve_thread_titles() or {}

    # Use consistent string keys for conversation titles.
    st.session_state["thread_titles"] = {
        str(thread_id): title
        for thread_id, title in titles.items()
    }

add_thread(st.session_state["thread_id"])


# ========================= Sidebar UI =========================

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

# Show the most recently used conversation first.
for thread_id in reversed(st.session_state["chat_threads"]):
    thread_key = str(thread_id)

    title = st.session_state["thread_titles"].get(
        thread_key, "New chat"
    )

    if st.sidebar.button(
        title,
        key=f"thread_{thread_key}",
        use_container_width=True,
    ):
        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        st.session_state["message_history"] = (
            reconstruct_message_history(messages)
        )

        st.rerun()


# ========================= Main Chat UI =========================

st.title("LangGraph Chatbot")

# Render existing conversation history.
for message in st.session_state["message_history"]:

    if message.get("tool_result"):
        with st.chat_message("assistant"):
            with st.expander(
                f"🛠️ {message['tool_name']} result",
                expanded=False,
            ):
                render_tool_result(message["content"])

    else:
        content = message.get("content", "")

        if content:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.text(content)
                else:
                    st.markdown(content)


# ========================= User Input =========================

user_input = st.chat_input("Type here...")

if user_input:
    thread_id = st.session_state["thread_id"]
    thread_key = str(thread_id)

    # Create a title from the first user message.
    if thread_key not in st.session_state["thread_titles"]:
        title = make_thread_title(user_input)

        st.session_state["thread_titles"][thread_key] = title

        save_thread_title(thread_key, title)

    # Move the active conversation to the top of the internal list.
    st.session_state["chat_threads"] = [
        existing_id
        for existing_id in st.session_state["chat_threads"]
        if str(existing_id) != thread_key
    ]

    st.session_state["chat_threads"].insert(0, thread_id)

    # Save and display the user's message.
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.text(user_input)

    # LangGraph execution configuration.
    CONFIG = {
        "configurable": {
            "thread_id": thread_id,
        },
        "metadata": {
            "thread_id": thread_key,
        },
        "run_name": "chat_turn",
    }

    # ========================= Assistant Streaming =========================

    with st.chat_message("assistant"):
        tool_results = st.container()
        response_placeholder = st.empty()

        ai_message_parts = []
        status_box = [None]

        def update_tool_status(label, state="running", expanded=True):
            """Create or update the tool execution status."""
            if status_box[0] is None:
                status_box[0] = st.status(
                    label,
                    state=state,
                    expanded=expanded,
                )
            else:
                status_box[0].update(
                    label=label,
                    state=state,
                    expanded=expanded,
                )

        # Stream messages from LangGraph.
        for message_chunk, metadata in chatbot.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=CONFIG,
            stream_mode="messages",
        ):

            # Handle tool results separately from assistant text.
            if isinstance(message_chunk, ToolMessage):
                tool_name = message_chunk.name or "Tool"

                update_tool_status(
                    f"✅ {tool_name} finished",
                    state="complete",
                    expanded=False,
                )

                tool_result = {
                    "role": "assistant",
                    "tool_result": True,
                    "tool_name": tool_name,
                    "content": message_chunk.content,
                }

                st.session_state["message_history"].append(
                    tool_result
                )

                with tool_results:
                    with st.expander(
                        f"🛠️ {tool_name} result",
                        expanded=True,
                    ):
                        render_tool_result(message_chunk.content)

                continue

            # Detect assistant tool calls to show execution status.
            if isinstance(message_chunk, AIMessage):
                tool_name = get_tool_name(message_chunk)

                if tool_name:
                    update_tool_status(
                        f"🔧 Using {tool_name}...",
                        state="running",
                        expanded=True,
                    )

                # Stream only assistant text.
                text = extract_text(message_chunk.content)

                if text:
                    ai_message_parts.append(text)

                    response_placeholder.markdown(
                        "".join(ai_message_parts)
                    )

        # Finalize the status indicator after streaming completes.
        if status_box[0] is not None:
            status_box[0].update(
                label="✅ Tool execution finished",
                state="complete",
                expanded=False,
            )

    # Save the completed assistant response.
    ai_message = "".join(ai_message_parts)

    if ai_message:
        st.session_state["message_history"].append({
            "role": "assistant",
            "content": ai_message,
        })