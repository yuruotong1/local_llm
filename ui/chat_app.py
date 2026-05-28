import streamlit as st
from app.model_runtime import generate_reply, get_runtime_info

st.set_page_config(page_title="Qwen Chat", page_icon="🤖")
st.title("Qwen Chat UI")

info = get_runtime_info()
st.caption(f"模型: {info['model']} | 设备: {info['device']}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            if message.get("thinking"):
                with st.expander("💭 思考过程 (Thinking Process)"):
                    st.write(message["thinking"])
            st.markdown(message["content"])
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("请输入问题"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("模型思考中..."):
            result = generate_reply(st.session_state.messages, max_new_tokens=512)
            if result["thinking"]:
                with st.expander("💭 思考过程 (Thinking Process)"):
                    st.write(result["thinking"])
            st.markdown(result["answer"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "thinking": result["thinking"],
        }
    )
