# ⚖️ Legal Document Analyzer
https://github.com/Santoshsharm07/Legal_Document_Analyzer/blob/main/image.png?raw=true

A complete end-to-end project for **summarizing legal documents** and **extracting key entities (persons, organizations, locations, etc.)**.  

It supports:
- 📄 Uploading PDFs (both **text PDFs** and **scanned PDFs**)  
- 🔍 OCR fallback via **Tesseract OCR** for scanned images  
- ✂️ **Hierarchical summarization** (splits long documents into chunks, then summarizes)  
- 🏷 **Named Entity Recognition (NER)** using a fine-tuned BERT model  
- 🖥 Streamlit frontend (upload multiple PDFs, view summaries + entities, download CSV/JSON)  
- 🛠 CLI tools for **batch processing directories** or **single PDFs**

---

## 🚀 Features

- **Frontend (Streamlit)**:  
  - Upload multiple PDFs at once  
  - Choose summarization model (fast vs accurate)  
  - Enable OCR if PDFs are scans  
  - Download results as CSV/JSON  

- **Backend (Python / PyTorch / Hugging Face Transformers)**:  
  - Summarization with `facebook/bart-large-cnn` or `sshleifer/distilbart-cnn-12-6`  
  - NER with `dslim/bert-base-NER` (aggregated entities)  
  - OCR fallback using Tesseract  

- **CLI Tools**:  
  - `analysis_cli.py` → Analyze a folder of PDFs → Export CSV  
  - `bert-ocr.py` → Analyze a single PDF with OCR → Export TXT  

---

## 📂 Project Structure

```
legal-doc-analyzer/
├── requirements.txt          # Python dependencies
├── legal_nlp.py              # Core NLP functions (summarization, OCR, NER)
├── streamlit_app.py          # Frontend (Streamlit UI)
├── analysis_cli.py           # CLI: batch process PDFs in a folder
├── bert-ocr.py               # CLI: single-PDF analysis with OCR fallback
└── README.md                 # Project documentation
```

---

## 🛠 Installation

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/legal-doc-analyzer.git
cd legal-doc-analyzer
```

### 2. Create a virtual environment  
Windows (PowerShell):
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Install **Tesseract OCR** (for scanned PDFs)

- **Windows**:  
  - Download from [UB Mannheim Builds](https://github.com/UB-Mannheim/tesseract/wiki)  
  - Install to:  
    ```
    C:\Program Files\Tesseract-OCR\tesseract.exe
    ```
  - Add to PATH or set env variable:  
    ```powershell
    setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
    ```

- **macOS**:
  ```bash
  brew install tesseract
  export TESSERACT_CMD="/opt/homebrew/bin/tesseract"  # for Apple Silicon
  ```

- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt-get update
  sudo apt-get install tesseract-ocr
  ```

✅ Verify installation:
```bash
tesseract --version
```

---

## 🖥 Running the Frontend

Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

- Upload one or more PDFs  
- Choose summarizer model & settings in the sidebar  
- View **summary + extracted entities**  
- Export results as CSV/JSON  

---

## ⚡ Running via CLI

### 1. Batch process a folder of PDFs
```bash
python analysis_cli.py --dir ./my_pdfs
# add --no-ocr if you don’t need OCR
```

→ Outputs a `legal_document_analysis.csv` in that folder.

### 2. Analyze a single PDF
```bash
python bert-ocr.py --pdf ./my_pdfs/contract.pdf
# add --no-ocr if you don’t need OCR
```

→ Outputs a text file in `./legal_document_analysis/`.

---

## 📊 Example Output

**Summary**:
```
The agreement establishes obligations between both parties regarding confidentiality and intellectual property. Termination clauses are defined, with jurisdiction in New Delhi courts.
```

**Named Entities**:
```
PER: John Doe, Jane Smith
ORG: Supreme Court of India, ABC Pvt Ltd
LOC: New Delhi
MISC: Intellectual Property Act
```

**CSV Export**:
| File         | Summary       | PER          | ORG                | LOC      | MISC                     |
|--------------|--------------|--------------|--------------------|----------|--------------------------|
| contract.pdf | The agreem…  | John Doe…    | Supreme Court…     | New Delhi| Intellectual Property…  |

---

## 🧠 Tech Stack

- **Python 3.9+**  
- **PyTorch** + **Hugging Face Transformers** (Summarization + NER)  
- **Streamlit** (frontend UI)  
- **PyMuPDF (`fitz`)** (PDF text extraction)  
- **Tesseract OCR** (image → text)  
- **Pandas** (CSV/JSON exports)  

---

## 📌 Notes

- Summarization models can be slow on CPU. For large docs, GPU recommended.  
- Default summarizer is `sshleifer/distilbart-cnn-12-6` (fast). You can switch to `facebook/bart-large-cnn` for better quality.  
- OCR adds time; disable it (`--no-ocr`) for digital PDFs.  

---

 
