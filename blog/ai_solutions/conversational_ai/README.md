# Conversational AI Demo (Streamlit + Transformers)

This directory contains a minimal working conversational AI demo built with Streamlit and Hugging Face Transformers. It generates assistant replies using a lightweight text-generation pipeline and is designed to be upgraded later for a multi-demo UI on Hugging Face Spaces.

## Quick start

- Install Python 3.9+ and run:
  ```bash
  pip install -U streamlit transformers torch --index-url https://download.pytorch.org/whl/cpu
  streamlit run app.py
  ```
- The app defaults to `distilbert/distilgpt2` (fallback: `gpt2`) via `pipeline("text-generation")`.

## Architecture

```
+---------------------+        +-----------------------------+
|  Streamlit Frontend | <----> |  Chat Controller (Session)  |
+----------+----------+        +---------------+-------------+
           |                                   |
           v                                   v
   User input text                     Prompt builder (history)
           |                                   |
           v                                   v
+----------+----------+        +-----------------------------+
|  HF Transformers    |        |  Text-Generation Pipeline   |
|  (distilgpt2/gpt2)  | -----> |  generate next assistant    |
+---------------------+        +-----------------------------+
```

- Stateless generation: we build a prompt from recent conversation history and sample next tokens.
- Replace the generator with an instruction-tuned chat model when deploying (e.g., Phi-3-mini or Llama variants) and/or use `transformers` chat templates.

## Files

- `app.py`: Streamlit app with a cached Transformers pipeline and simple chat loop.
- `README.md`: This file with setup, architecture, and links.

## Model and docs links

- Model card: https://huggingface.co/distilbert/distilgpt2
- Transformers docs (pipeline): https://huggingface.co/docs/transformers/main_classes/pipelines
- Streamlit docs: https://docs.streamlit.io/

## Notes for future Hugging Face Space integration

- Convert this single app into a multi-demo launcher using `st.sidebar.selectbox` or `st.tabs()`.
- Add a `requirements.txt` to pin versions and ensure CPU-friendly wheels on Spaces.
- Consider using `transformers` chat-oriented models and `TextIteratorStreamer` for streaming tokens.
- Add safety/mode filters, system prompts, and conversation management utilities.
