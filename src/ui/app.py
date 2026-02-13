import os
import sys
import time
import streamlit as st
from langchain_core.messages import HumanMessage

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.agent.graph import build_graph, get_next_session_id

def app() -> None:
    st.set_page_config(
        page_title="MedGemma Local Triage",
        page_icon="🏥",
        layout="wide"
    )

    if "agent" not in st.session_state:
        st.session_state.agent = build_graph()
    if "session_id" not in st.session_state:
        st.session_state.session_id = get_next_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.title("🏥 System Dashboard")
        st.success("Backend: MedGemma 1.5 (Connected)")
        st.info(f"Session ID: {st.session_state.session_id}")
        
        st.divider()
        st.markdown("### 🔒 Privacy & Security")
        st.caption("This system utilizes an Edge AI architecture. All inference and retrieval are performed locally. No data is uploaded to the cloud.")
        
        if st.button("Start New Session"):
            st.session_state.messages = []
            st.session_state.session_id = get_next_session_id()
            st.rerun()

    st.title("MedGemma: Intelligent Triage & Clinical Navigation")
    st.markdown("---")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Describe your symptoms or medical emergency..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.status("🩺 Analyzing symptoms and retrieving guidelines...", expanded=True) as status:
                config = {"configurable": {"thread_id": st.session_state.session_id}}
                input_data = {"messages": [HumanMessage(content=user_input)]}
                
                result = st.session_state.agent.invoke(input_data, config=config)
                
                severity = result.get("severity", "NORMAL")
                raw_answer = result["messages"][-1].content
                if "1." in raw_answer:
                    final_answer = "1." + raw_answer.split("1.", 1)[1]
                else:
                    final_answer = raw_answer
                
                icon = "🚨" if severity == "CRITICAL" else "🩺"
                status.update(label=f"{icon} Triage Complete: {severity} Level", state="complete")

            def stream_data():
                for word in final_answer.split(" "):
                    yield word + " "
                    time.sleep(0.05) 

            full_response = st.write_stream(stream_data)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    app()