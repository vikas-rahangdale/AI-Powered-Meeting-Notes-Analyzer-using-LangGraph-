# Supreme
import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END , START
import tiktoken

# --- Load API key from .env ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found in .env file. Please add it to your .env file.")
    st.stop()

# Initialize LLM
llm = ChatOpenAI(model="gpt-5.6-luna", temperature=0, api_key=api_key)

# --- Tokenizer for GPT-5.6-Luna ---
encoding = tiktoken.encoding_for_model("gpt-5.6-luna")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def chunk_text(text: str, max_tokens: int = 1000) -> list:
    words = text.split()
    chunks, current_chunk = [], []
    current_len = 0
    for word in words:
        word_len = len(encoding.encode(word))
        if current_len + word_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_len = word_len
        else:
            current_chunk.append(word)
            current_len += word_len
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def run_llm(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content.strip()

def safe_transcript(transcript: str, max_context: int = 6000) -> str:
    tokens = count_tokens(transcript)
    if tokens <= max_context:
        return transcript
    else:
        # Summarize chunks
        chunks = chunk_text(transcript, max_tokens=1000)
        summaries = [run_llm(f"Summarize this chunk:\n{chunk}") for chunk in chunks]
        return " ".join(summaries)

# --- Agents ---
def input_agent(transcript: str) -> str:
    return transcript.strip()

def topic_agent(transcript: str) -> list:
    prompt = f"Extract the main discussion topics from this meeting transcript:\n{transcript}"
    return [t.strip("-• ") for t in run_llm(prompt).split("\n") if t.strip()]

def summary_agent(transcript: str) -> str:
    prompt = f"Summarize this meeting in 3-5 sentences:\n{transcript}"
    return run_llm(prompt)

def action_agent(transcript: str) -> list:
    prompt = f"""
    Extract action items from this transcript.
    Format each as: Task - Responsible (if not mentioned, write 'Responsible: Not specified').
    Transcript:
    {transcript}
    """
    return [a.strip("-• ") for a in run_llm(prompt).split("\n") if a.strip()]

def priority_agent(action_items: list, transcript: str) -> str:
    if not action_items or action_items == [""]:
        return "No action items identified in this meeting."
    prompt = f"""
    Based on urgency words (urgent, ASAP, by tomorrow, this week, before Friday, etc ) and context in the transcript,
    classify the overall priority of these tasks as High, Medium, or Low.
    Transcript: {transcript}
    """
    return run_llm(prompt)

# --- LangGraph Workflow ---
class MeetingState(dict):
    transcript: str
    topics: list
    summary: str
    action_items: list
    priority: str

workflow = StateGraph(MeetingState)

workflow.add_node("Input", lambda state: {"transcript": input_agent(state["transcript"])})
workflow.add_node("Topics", lambda state: {"topics": topic_agent(state["transcript"])})
workflow.add_node("Summary", lambda state: {"summary": summary_agent(state["transcript"])})
workflow.add_node("Actions", lambda state: {"action_items": action_agent(state["transcript"])})
workflow.add_node("Priority", lambda state: {"priority": priority_agent(state["action_items"], state["transcript"])})

workflow.set_entry_point("Input")
workflow.add_edge("Input", "Topics")
workflow.add_edge("Topics", "Summary")
workflow.add_edge("Summary", "Actions")
workflow.add_conditional_edges(
    "Actions",
    lambda state: "Priority" if state["action_items"] else END,
    {"Priority": "Priority", END: END}
)
workflow.add_edge("Priority", END)

meeting_notes_workflow = workflow.compile()

# --- Streamlit UI ---
st.title("📋 AI-Powered Meeting Notes Analyzer using LangGraph)")

st.write("Paste or upload a meeting transcript to generate structured notes automatically.")

# Input options
transcript_input = st.text_area("Meeting Transcript", height=200)
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

if uploaded_file is not None:
    transcript_input = uploaded_file.read().decode("utf-8")

if st.button("Analyze Meeting"):
    if transcript_input.strip():
        # Safeguard for large transcripts
        condensed_transcript = safe_transcript(transcript_input)

        result = meeting_notes_workflow.invoke({"transcript": condensed_transcript})

        # Display results
        st.subheader("Meeting Summary")
        st.write(result.get("summary", ""))

        st.subheader("Key Topics")
        for i, t in enumerate(result.get("topics", []), 1):
            st.write(f"{i}. {t}")

        st.subheader("Action Items")
        actions = result.get("action_items", [])
        if actions and actions != [""]:
            for i, a in enumerate(actions, 1):
                st.write(f"{i}. {a}")
        else:
            st.write("No action items identified.")

        st.subheader("Priority Level")
        st.write(result.get("priority", ""))

        # Download option
        st.download_button(
            label="⬇️ Download Notes as JSON",
            data=json.dumps(result, indent=2),
            file_name="meeting_notes.json",
            mime="application/json"
        )
    else:
        st.warning("⚠️ Please enter or upload a transcript before analyzing.")
