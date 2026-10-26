import json
import uuid
import streamlit as st
from langgraph_tool_backend import (
    chatbot,
    retrieve_all_threads,
    retrieve_thread_titles,
    save_thread_title,
)
from langchain_core.messages import HumanMessage, ToolMessage

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

def make_thread_title(message, max_length=32):
    title = " ".join(message.split())
    if len(title) > max_length:
        return title[:max_length - 3].rstrip() + "..."
    return title or "New chat"

def render_tool_result(content):
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

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str)
            else block.get("text", "")
            for block in content
            if isinstance(block, str)
            or (isinstance(block, dict) and isinstance(block.get("text"), str))
        )
    return ""

# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = retrieve_thread_titles()

add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads']:
    title = st.session_state['thread_titles'].get(thread_id, "New chat")
    if st.sidebar.button(title, key=f"thread_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_messages.append({'role': 'user', 'content': msg.content})
            elif isinstance(msg, ToolMessage):
                temp_messages.append({
                    'role': 'assistant',
                    'tool_result': True,
                    'tool_name': msg.name or 'Tool',
                    'content': msg.content,
                })
            elif extract_text(msg.content):
                temp_messages.append({
                    'role': 'assistant',
                    'content': extract_text(msg.content),
                })

        st.session_state['message_history'] = temp_messages


# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    if message.get('tool_result'):
        with st.chat_message('assistant'):
            with st.expander(f"{message['tool_name']} result", expanded=True):
                render_tool_result(message['content'])
    elif message['content']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    thread_id = st.session_state['thread_id']
    if thread_id not in st.session_state['thread_titles']:
        st.session_state['thread_titles'][thread_id] = make_thread_title(user_input)
    save_thread_title(
        str(thread_id),
        st.session_state['thread_titles'][thread_id],
    )
    st.session_state['chat_threads'] = [
        existing_thread
        for existing_thread in st.session_state['chat_threads']
        if str(existing_thread) != str(thread_id)
    ]
    st.session_state['chat_threads'].insert(0, thread_id)

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    CONFIG={
        "configurable": {"thread_id": st.session_state["thread_id"]},
         "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
         "run_name": "chat_turn"
    }






    with st.chat_message('assistant'):
        tool_results = st.container()
        response_placeholder = st.empty()
        ai_message_parts = []

        for message_chunk, _ in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode='messages'
        ):
            if getattr(message_chunk, 'type', None) == 'tool':
                tool_name = getattr(message_chunk, 'name', None) or 'Tool'
                tool_result = {
                    'role': 'assistant',
                    'tool_result': True,
                    'tool_name': tool_name,
                    'content': message_chunk.content,
                }
                st.session_state['message_history'].append(tool_result)
                with tool_results:
                    with st.expander(f"{tool_name} result", expanded=True):
                        render_tool_result(message_chunk.content)
                continue

            text = extract_text(message_chunk.content)
            if text:
                ai_message_parts.append(text)
                response_placeholder.markdown(''.join(ai_message_parts))

    ai_message = ''.join(ai_message_parts)
    if ai_message:
        st.session_state['message_history'].append({
            'role': 'assistant',
            'content': ai_message,
        })