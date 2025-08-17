
import os
import csv
import argparse
from legal_nlp import NLPModels, extract_text_from_pdf_path, summarize_long_text, extract_entities

def process_directory(directory_path: str, enable_ocr: bool = True, min_len: int = 60, max_len: int = 200, chunk_tokens: int = 700):
    models = NLPModels()
    rows = [["File name", "File Summary", "Persons", "Organizations", "Locations", "Misc"]]
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            text = extract_text_from_pdf_path(pdf_path, enable_ocr=enable_ocr)
            summary = summarize_long_text(text, models, min_len=min_len, max_len=max_len, chunk_tokens=chunk_tokens)
            ents = extract_entities(text, models)
            rows.append([
                filename,
                summary,
                ", ".join(ents.get("PER", [])),
                ", ".join(ents.get("ORG", [])),
                ", ".join(ents.get("LOC", [])),
                ", ".join(ents.get("MISC", [])),
            ])
    out_csv = os.path.join(directory_path, "legal_document_analysis.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch analyze legal PDFs in a directory.")
    parser.add_argument("--dir", required=True, help="Directory containing PDFs")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback")
    args = parser.parse_args()
    process_directory(args.dir, enable_ocr=not args.no_ocr)
