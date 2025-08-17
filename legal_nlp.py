
import re
import io
import os
from typing import List, Dict, Any

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification


def _maybe_configure_tesseract_from_env() -> None:
    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def extract_text_from_pdf_filelike(file_like, enable_ocr: bool = True, ocr_dpi: int = 200) -> str:
    _maybe_configure_tesseract_from_env()
    doc = fitz.open(stream=file_like.read(), filetype="pdf")
    text = ""
    for page in doc:
        page_text = page.get_text()
        if enable_ocr and not page_text.strip():
            # Render page to image and OCR it if the page has no extractable text
            pix = page.get_pixmap(dpi=ocr_dpi, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(img)
        text += page_text + "\n"
    return text


def extract_text_from_pdf_path(path: str, enable_ocr: bool = True, ocr_dpi: int = 200) -> str:
    with open(path, "rb") as f:
        return extract_text_from_pdf_filelike(f, enable_ocr=enable_ocr, ocr_dpi=ocr_dpi)


class NLPModels:
    """
    Holder for summarization + NER pipelines so they can be reused across documents.
    """
    def __init__(self, sum_model: str = "sshleifer/distilbart-cnn-12-6", ner_model: str = "dslim/bert-base-NER"):
        self.sum_name = sum_model
        self.tokenizer = AutoTokenizer.from_pretrained(sum_model)
        self.summarizer = pipeline(
            "summarization",
            model=AutoModelForSeq2SeqLM.from_pretrained(sum_model),
            tokenizer=self.tokenizer,
        )
        self.ner = pipeline(
            "ner",
            model=AutoModelForTokenClassification.from_pretrained(ner_model),
            tokenizer=AutoTokenizer.from_pretrained(ner_model),
            aggregation_strategy="simple",
        )


def _chunk_text_for_model(text: str, tokenizer: AutoTokenizer, max_tokens: int = 700) -> List[str]:
    """
    Split text by paragraphs while respecting a token budget per chunk.
    """
    paras = re.split(r"\n{2,}", text)
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        if not p.strip():
            continue
        tokens = tokenizer.encode(p, add_special_tokens=False)
        if cur_len + len(tokens) > max_tokens and cur:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_len = len(tokens)
        else:
            cur.append(p)
            cur_len += len(tokens)
    if cur:
        chunks.append("\n\n".join(cur))
    if not chunks:
        chunks = [text]
    return chunks


def summarize_long_text(
    text: str,
    models: NLPModels,
    min_len: int = 60,
    max_len: int = 200,
    chunk_tokens: int = 700,
) -> str:
    """
    Hierarchical/recursive summarization for long documents.
    """
    text = text.strip()
    if not text:
        return ""
    # Simple path
    if len(models.tokenizer.encode(text, add_special_tokens=False)) <= chunk_tokens:
        out = models.summarizer(text, min_length=min_len, max_length=max_len, do_sample=False)
        return out[0]["summary_text"]

    # Chunked path
    chunks = _chunk_text_for_model(text, models.tokenizer, max_tokens=chunk_tokens)
    partials = [
        models.summarizer(c, min_length=min_len, max_length=max_len, do_sample=False)[0]["summary_text"]
        for c in chunks
    ]
    combined = " ".join(partials).strip()
    if not combined:
        return ""
    # If combined is still long, run one more pass
    if len(models.tokenizer.encode(combined, add_special_tokens=False)) > chunk_tokens:
        out = models.summarizer(combined, min_length=min_len, max_length=max_len, do_sample=False)
        return out[0]["summary_text"]
    return combined


def extract_entities(text: str, models: NLPModels) -> Dict[str, Any]:
    """
    Extract PER/ORG/LOC/MISC using a token-classification model with simple aggregation.
    """
    text = text.strip()
    if not text:
        return {"PER": [], "ORG": [], "LOC": [], "MISC": []}
    ents = models.ner(text)
    grouped = {"PER": set(), "ORG": set(), "LOC": set(), "MISC": set()}
    for e in ents:
        grp = e.get("entity_group", "MISC")
        word = e.get("word", "").strip()
        if not word:
            continue
        if grp in grouped:
            grouped[grp].add(word)
        else:
            grouped["MISC"].add(word)
    return {k: sorted(v) for k, v in grouped.items()}
