
import os
import argparse
from legal_nlp import NLPModels, extract_text_from_pdf_path, summarize_long_text, extract_entities

def process(pdf_path: str, min_len: int = 60, max_len: int = 200, chunk_tokens: int = 700, enable_ocr: bool = True):
    models = NLPModels()
    text = extract_text_from_pdf_path(pdf_path, enable_ocr=enable_ocr)
    summary = summarize_long_text(text, models, min_len=min_len, max_len=max_len, chunk_tokens=chunk_tokens)
    ents = extract_entities(text, models)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join(os.path.dirname(pdf_path), "legal_document_analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_analysis.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Summary:\n")
        f.write(summary)
        f.write("\n\nNamed Entities:\n")
        for k in ["PER", "ORG", "LOC", "MISC"]:
            vals = ents.get(k, [])
            f.write(f"{k}: {', '.join(vals)}\n")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR-aware single PDF analyzer")
    parser.add_argument("--pdf", required=True, help="Path to a PDF")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback")
    args = parser.parse_args()
    process(args.pdf, enable_ocr=not args.no_ocr)
