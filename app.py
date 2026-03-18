import requests
import streamlit as st


st.set_page_config(page_title="My AI Chat", layout="wide")
st.title("My AI Chat")

token = st.secrets.get("HF_TOKEN", "").strip()
api_url = "https://router.huggingface.co/v1/chat/completions"

if "messages" not in st.session_state:
    st.session_state.messages = []

if not token:
    st.error("Missing Hugging Face token. Please set HF_TOKEN in .streamlit/secrets.toml.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        payload = {
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "messages": st.session_state.messages,
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
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_reply}
                )
                with st.chat_message("assistant"):
                    st.write(assistant_reply)
            else:
                st.error(f"API error {response.status_code}: {response.text}")
        except requests.RequestException as exc:
            st.error(f"Network error: {exc}")
