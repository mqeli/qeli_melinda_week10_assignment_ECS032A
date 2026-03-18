from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import re
import time
import requests
import streamlit as st


st.set_page_config(page_title="My AI Chat", layout="wide")
st.title("My AI Chat")

token = st.secrets.get("HF_TOKEN", "").strip()
api_url = "https://router.huggingface.co/v1/chat/completions"

base_dir = Path(__file__).resolve().parent
chats_dir = base_dir / "chats"
chats_dir.mkdir(parents=True, exist_ok=True)
memory_path = base_dir / "memory.json"

if "chats" not in st.session_state:
    st.session_state.chats = []
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "next_chat_id" not in st.session_state:
    st.session_state.next_chat_id = 1
if "chats_loaded" not in st.session_state:
    st.session_state.chats_loaded = False
if "memory" not in st.session_state:
    st.session_state.memory = {}


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


def load_memory(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_memory(path: Path, memory: dict) -> None:
    path.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def merge_memory(existing: dict, updates: dict) -> dict:
    merged = dict(existing)
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value
    return merged


def extract_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

if not st.session_state.chats_loaded:
    st.session_state.chats = load_chats_from_disk()
    max_id = max((chat["id"] for chat in st.session_state.chats), default=0)
    st.session_state.next_chat_id = max_id + 1
    if st.session_state.active_chat_id is None and st.session_state.chats:
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]
    st.session_state.chats_loaded = True

if not st.session_state.memory:
    st.session_state.memory = load_memory(memory_path)

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

    with st.expander("User Memory", expanded=False):
        if st.session_state.memory:
            st.json(st.session_state.memory)
        else:
            st.write("No memory stored yet.")
        if st.button("Clear Memory"):
            st.session_state.memory = {}
            save_memory(memory_path, st.session_state.memory)
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
                "messages": (
                    [
                        {
                            "role": "system",
                            "content": (
                                "Use the following user memory to personalize responses: "
                                f"{json.dumps(st.session_state.memory)}"
                            ),
                        }
                    ]
                    if st.session_state.memory
                    else []
                )
                + active_chat["messages"],
                "max_tokens": 200,
                "stream": True,
            }

            try:
                response = requests.post(
                    api_url,
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                    timeout=60,
                    stream=True,
                )
                if response.status_code == 200:
                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        assembled = ""
                        for line in response.iter_lines(decode_unicode=True):
                            if not line:
                                continue
                            if line.startswith("data: "):
                                chunk = line.replace("data: ", "", 1).strip()
                                if chunk == "[DONE]":
                                    break
                                try:
                                    data = json.loads(chunk)
                                except json.JSONDecodeError:
                                    continue
                                choices = data.get("choices", [])
                                if not choices:
                                    continue
                                delta = choices[0].get("delta", {}).get("content", "")
                                if delta:
                                    assembled += delta
                                    placeholder.write(assembled)
                                    time.sleep(0.02)

                        if assembled.strip():
                            active_chat["messages"].append(
                                {"role": "assistant", "content": assembled.strip()}
                            )
                            save_chat(active_chat)

                            # Extract user traits/preferences from the latest user message.
                            extract_prompt = (
                                "Given this user message, extract any personal facts or "
                                "preferences as a JSON object. If none, return {}. "
                                "Respond with JSON only (no extra text).\n\n"
                                f"User message: {user_input}"
                            )
                            extract_payload = {
                                "model": "meta-llama/Llama-3.2-1B-Instruct",
                                "messages": [{"role": "user", "content": extract_prompt}],
                                "max_tokens": 120,
                            }
                            try:
                                extract_response = requests.post(
                                    api_url,
                                    headers={"Authorization": f"Bearer {token}"},
                                    json=extract_payload,
                                    timeout=60,
                                )
                                if extract_response.status_code == 200:
                                    extract_data = extract_response.json()
                                    extract_text = (
                                        extract_data["choices"][0]["message"]["content"].strip()
                                    )
                                    extracted = extract_json_object(extract_text)
                                    if isinstance(extracted, dict) and extracted:
                                        st.session_state.memory = merge_memory(
                                            st.session_state.memory, extracted
                                        )
                                        save_memory(memory_path, st.session_state.memory)
                                else:
                                    st.warning(
                                        f"Memory extraction error {extract_response.status_code}"
                                    )
                            except requests.RequestException:
                                st.warning("Memory extraction request failed.")
                else:
                    st.error(f"API error {response.status_code}: {response.text}")
            except requests.RequestException as exc:
                st.error(f"Network error: {exc}")
