import streamlit as st
import ollama

# 初始化 Ollama 客户端，连接本地服务
client = ollama.Client(host="http://localhost:11434")

# 初始化会话状态，保存对话历史
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# 设置页面标题
st.title("🤖 宝绿特大模型聊天机器人")

# 显示历史对话
for msg in st.session_state['messages']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# 获取用户输入
prompt = st.chat_input("请输入你的问题...")

if prompt:
    # 添加用户消息到历史记录并显示
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用模型生成回复，显示加载状态
    with st.spinner("AI思考中..."):
        try:
            response = client.chat(
                model="llama3:8b-instruct-q4_K_M",  # 使用你下载的模型名称
                messages=st.session_state['messages']
            )
            ai_response = response['message']['content']  # 正确提取 AI 回复内容

            # 添加 AI 回复到历史记录并显示
            st.session_state['messages'].append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.markdown(ai_response)
        except Exception as e:
            st.error(f"调用模型失败: {e}")
