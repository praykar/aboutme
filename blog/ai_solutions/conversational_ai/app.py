import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Conversational AI Demo", page_icon="🤖")

@st.cache_resource(show_spinner=False)
def get_pipeline():
    # Small, CPU-friendly chat model fallback using text-generation
    # You can change to another model ID compatible with pipeline("text-generation")
    try:
        generator = pipeline("text-generation", model="distilbert/distilgpt2")
    except Exception:
        # Fallback to default gpt2 if distilgpt2 is unavailable in environment
        generator = pipeline("text-generation", model="gpt2")
    return generator

st.title("🤖 Conversational AI Demo (Streamlit + Transformers)")
st.caption("Minimal working demo using Hugging Face Transformers pipeline. Not stateful, illustrative only.")

if "history" not in st.session_state:
    st.session_state.history = []

gen = get_pipeline()

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state.history.append(("user", user_input.strip()))
    prompt = "\n\n".join([f"User: {u}" if r=="user" else f"Assistant: {u}" for r,u in st.session_state.history])
    prompt += "\nAssistant:"
    out = gen(prompt, max_new_tokens=120, do_sample=True, top_p=0.92, temperature=0.7, num_return_sequences=1)[0]["generated_text"]
    # Extract only the new assistant part after the last 'Assistant:' tag
    reply = out.split("Assistant:")[-1].strip().split("User:")[0].strip()
    st.session_state.history.append(("assistant", reply if reply else "I'm here to help!"))

st.subheader("Conversation")
for role, text in st.session_state.history[-12:]:
    if role == "user":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Assistant:** {text}")

st.sidebar.header("About")
st.sidebar.markdown(
    "- Model: distilbert/distilgpt2 (fallback: gpt2)\n"
    "- Pipeline: text-generation\n"
    "- Purpose: Minimal local demo for future HF Space integration"
)

st.sidebar.header("Notes")
st.sidebar.markdown(
    "- This simple demo builds a prompt from chat history and generates the next response.\n"
    "- Replace with chat-optimized models (e.g., microsoft/Phi-3-mini-4k-instruct) when available.\n"
    "- Add moderation, safety filters, and better formatting for production."
)
