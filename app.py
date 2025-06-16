# app.py
import streamlit as st
import pandas as pd
import json
from etl_engine import execute_etl_step, summarize_etl_step
from llm_parser import parse_etl_instruction

st.set_page_config(page_title="ETL Chatbot", layout="wide", page_icon="🤖")

# ---------- SIDEBAR ----------
st.sidebar.title("📂 ETL Chatbot")
st.sidebar.write("Automate data cleaning with natural language.")

uploaded_file = st.sidebar.file_uploader("Upload your dataset (.csv)", type=["csv"])
if uploaded_file and "file_uploaded" not in st.session_state:
    try:
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        st.session_state["original_df"] = df.copy()
        st.session_state["df"] = df
        st.session_state["file_uploaded"] = True
        st.sidebar.success("✅ File loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Upload failed: {e}")

# ---------- MAIN TITLE ----------
st.markdown("<h1 style='text-align:center;'>🤖 ETL Automation Chatbot</h1>", unsafe_allow_html=True)

# ---------- SESSION INIT ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None

# ---------- DATA PREVIEW ----------
if st.session_state.df is not None:
    tabs = st.tabs(["📄 Original Dataset", "📊 Processed Dataset"])
    with tabs[0]:
        st.dataframe(st.session_state["original_df"].head(20), use_container_width=True)
    with tabs[1]:
        st.dataframe(st.session_state.df.head(20), use_container_width=True)

# ---------- CHAT HISTORY DISPLAY ----------
st.markdown("---")
chat_container = st.container()
for msg in st.session_state.messages:
    with chat_container:
        if msg["role"] == "user":
            st.markdown(f"🧑 **You**:\n\n{msg['content']}")
        else:
            st.markdown(f"🤖 **Bot**:\n\n{msg['content']}")

# ---------- CHAT INPUT ----------
user_input = st.chat_input("Type your ETL instruction here...")

if user_input:
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("🤖 Thinking..."):
            try:
                steps = parse_etl_instruction(user_input)
                df = st.session_state.df.copy()

                summaries = []
                if isinstance(steps, list):
                    for s in steps:
                        df = execute_etl_step(df, s)
                        summaries.append(summarize_etl_step(s))
                else:
                    df = execute_etl_step(df, steps)
                    summaries.append(summarize_etl_step(steps))

                # Update processed dataset and save
                st.session_state.df = df
                df.to_csv("processed_output.txt", sep="\t", index=False)

                # Show response
                summary_text = "\n".join([f"- 🛠️ {s}" for s in summaries])
                reply = f"✅ **Applied ETL Steps:**\n\n{summary_text}"
                st.session_state.messages.append({"role": "assistant", "content": reply})

            except Exception as e:
                error_msg = f"❌ Error: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

        # 🔄 Trigger rerun AFTER all updates
        st.rerun()
