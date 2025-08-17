
import os
import json
import time
import streamlit as st

from legal_nlp import (
    NLPModels,
    extract_text_from_pdf_filelike,
    summarize_long_text,
    extract_entities,
)

st.set_page_config(page_title="Legal Document Analyzer", layout="wide")

st.title("⚖️ Legal Document Analyzer")
st.caption("Summarize PDFs and extract named entities (PER/ORG/LOC/MISC).")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Summarization model",
        options=[
            "sshleifer/distilbart-cnn-12-6",
            "facebook/bart-large-cnn",
        ],
        index=0,
        help="Smaller model loads faster; large model may give better quality.",
    )
    min_len = st.slider("Summary min length", 10, 200, 60, 5)
    max_len = st.slider("Summary max length", 50, 400, 200, 10)
    chunk_tokens = st.slider("Chunk size (tokens)", 300, 900, 700, 50)
    enable_ocr = st.checkbox("Enable OCR fallback (slower)", value=True)
    tess_cmd = st.text_input(
        "Path to Tesseract (optional)",
        value=os.getenv("TESSERACT_CMD", ""),
        help="Windows example: C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
    )
    if tess_cmd:
        os.environ["TESSERACT_CMD"] = tess_cmd

@st.cache_resource(show_spinner=True)
def load_models(selected_model: str) -> NLPModels:
    return NLPModels(sum_model=selected_model, ner_model="dslim/bert-base-NER")

models = load_models(model_name)

uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

results = []
if uploaded_files:
    for f in uploaded_files:
        with st.spinner(f"Processing {f.name} ..."):
            text = extract_text_from_pdf_filelike(f, enable_ocr=enable_ocr)
            summary = summarize_long_text(text, models, min_len=min_len, max_len=max_len, chunk_tokens=chunk_tokens)
            ents = extract_entities(text, models)
        with st.expander(f"📄 {f.name}", expanded=True):
            st.subheader("Summary")
            st.write(summary if summary else "_(empty)_")
            st.subheader("Named Entities")
            col1, col2, col3, col4 = st.columns(4)
            col1.write({"PER": ents.get("PER", [])})
            col2.write({"ORG": ents.get("ORG", [])})
            col3.write({"LOC": ents.get("LOC", [])})
            col4.write({"MISC": ents.get("MISC", [])})
        results.append(
            {
                "file": f.name,
                "summary": summary,
                "PER": ", ".join(ents.get("PER", [])),
                "ORG": ", ".join(ents.get("ORG", [])),
                "LOC": ", ".join(ents.get("LOC", [])),
                "MISC": ", ".join(ents.get("MISC", [])),
            }
        )

if results:
    st.divider()
    st.subheader("Batch Results")
    # JSON download
    json_bytes = json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("Download JSON", data=json_bytes, file_name="legal_analysis.json", mime="application/json")
    # CSV download
    import pandas as pd
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="legal_analysis.csv", mime="text/csv")

st.info("Tip: If a page contains scanned images only, enable OCR and set the correct Tesseract path in the sidebar.")
