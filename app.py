from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import requests
import streamlit as st


st.set_page_config(page_title="My AI Chat", layout="wide")
st.title("My AI Chat")

token = st.secrets.get("HF_TOKEN", "").strip()
api_url = "https://router.huggingface.co/v1/chat/completions"

base_dir = Path(__file__).resolve().parent
chats_dir = base_dir / "chats"
chats_dir.mkdir(parents=True, exist_ok=True)

if "chats" not in st.session_state:
    st.session_state.chats = []
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "next_chat_id" not in st.session_state:
    st.session_state.next_chat_id = 1
if "chats_loaded" not in st.session_state:
    st.session_state.chats_loaded = False


def chat_file_path(chat_id: int) -> Path:
    return chats_dir / f"chat_{chat_id}.json"


def load_chats_from_disk() -> list[dict]:
    chats: list[dict] = []
    for path in sorted(chats_dir.glob("chat_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        chat_id = data.get("id")
        if not isinstance(chat_id, int):
            continue
        chats.append(
            {
                "id": chat_id,
                "title": data.get("title", "New Chat"),
                "created_at": data.get("created_at", ""),
                "messages": data.get("messages", []),
            }
        )
    return chats


def save_chat(chat: dict) -> None:
    payload = {
        "id": chat["id"],
        "title": chat.get("title", "New Chat"),
        "created_at": chat.get("created_at", ""),
        "messages": chat.get("messages", []),
    }
    chat_file_path(chat["id"]).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def create_chat() -> int:
    chat_id = st.session_state.next_chat_id
    st.session_state.next_chat_id += 1
    chat = {
        "id": chat_id,
        "title": "New Chat",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": [],
    }
    st.session_state.chats.append(chat)
    save_chat(chat)
    return chat_id


def get_active_chat() -> Optional[dict]:
    for chat in st.session_state.chats:
        if chat["id"] == st.session_state.active_chat_id:
            return chat
    return None


if not st.session_state.chats_loaded:
    st.session_state.chats = load_chats_from_disk()
    max_id = max((chat["id"] for chat in st.session_state.chats), default=0)
    st.session_state.next_chat_id = max_id + 1
    if st.session_state.active_chat_id is None and st.session_state.chats:
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]
    st.session_state.chats_loaded = True

with st.sidebar:
    st.subheader("Chats")
    if st.button("New Chat"):
        st.session_state.active_chat_id = create_chat()
        st.rerun()

    chat_list = st.container(height=350, border=True)
    with chat_list:
        for chat in st.session_state.chats:
            is_active = chat["id"] == st.session_state.active_chat_id
            label_prefix = "▶ " if is_active else ""
            with st.container():
                cols = st.columns([8, 2])
                if cols[0].button(
                    f"{label_prefix}{chat['title']} ({chat['created_at']})",
                    key=f"chat_{chat['id']}",
                ):
                    st.session_state.active_chat_id = chat["id"]
                    st.rerun()
                if cols[1].button("✕", key=f"del_{chat['id']}"):
                    st.session_state.chats = [
                        c for c in st.session_state.chats if c["id"] != chat["id"]
                    ]
                    try:
                        chat_file_path(chat["id"]).unlink(missing_ok=True)
                    except OSError:
                        pass
                    if st.session_state.active_chat_id == chat["id"]:
                        st.session_state.active_chat_id = (
                            st.session_state.chats[0]["id"]
                            if st.session_state.chats
                            else None
                        )
                    st.rerun()

if not token:
    st.error("Missing Hugging Face token. Please set HF_TOKEN in .streamlit/secrets.toml.")
else:
    active_chat = get_active_chat()
    if active_chat is None:
        st.info("No active chat. Create a new chat from the sidebar.")
    else:
        for message in active_chat["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_input = st.chat_input("Type your message...")
        if user_input:
            active_chat["messages"].append({"role": "user", "content": user_input})
            if active_chat["title"] == "New Chat":
                active_chat["title"] = user_input[:40]
            save_chat(active_chat)

            with st.chat_message("user"):
                st.write(user_input)

            payload = {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "messages": active_chat["messages"],
                "max_tokens": 200,
            }

            try:
                response = requests.post(
                    api_url,
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                    timeout=60,
                )
                if response.status_code == 200:
                    data = response.json()
                    assistant_reply = data["choices"][0]["message"]["content"].strip()
                    active_chat["messages"].append(
                        {"role": "assistant", "content": assistant_reply}
                    )
                    save_chat(active_chat)
                    with st.chat_message("assistant"):
                        st.write(assistant_reply)
                else:
                    st.error(f"API error {response.status_code}: {response.text}")
            except requests.RequestException as exc:
                st.error(f"Network error: {exc}")
