# PDF Document Summarizer Skill

## Overview
This skill extracts text from PDF documents and generates structured, intelligent summaries using AI. It supports single or multiple PDF files and produces summaries at different levels of detail.

## Features
- **EXTRACT** text from PDFs (including multi-page documents)
- **SUMMARIZE** content using OpenAI GPT models
- **SUPPORT** multiple summary modes: brief, standard, and detailed
- **HANDLE** scanned PDFs with OCR fallback (via pytesseract)
- **OUTPUT** results as plain text or structured JSON
- **BATCH PROCESS** multiple PDFs in one run

## Requirements

### Python Dependencies
```
openai>=1.0.0
PyMuPDF>=1.23.0
pytesseract>=0.3.10
Pillow>=10.0.0
python-dotenv>=1.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

### Environment Variables
Set the following in a `.env` file or export them:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Basic Usage
```bash
python main.py --pdf path/to/document.pdf
```

### Specify Summary Mode
```bash
python main.py --pdf path/to/document.pdf --mode brief
python main.py --pdf path/to/document.pdf --mode standard
python main.py --pdf path/to/document.pdf --mode detailed
```

### Batch Processing Multiple PDFs
```bash
python main.py --pdf doc1.pdf doc2.pdf doc3.pdf --mode standard
```

### Save Output to JSON
```bash
python main.py --pdf path/to/document.pdf --output results.json
```

### Full Options
```bash
python main.py --pdf <path(s)> [--mode brief|standard|detailed] [--output <file.json>] [--max-pages <int>]
```

| Argument      | Description                                      | Default    |
|---------------|--------------------------------------------------|------------|
| `--pdf`       | Path(s) to PDF file(s)                           | *required* |
| `--mode`      | Summary mode: brief, standard, detailed          | standard   |
| `--output`    | Save results to a JSON file                      | None       |
| `--max-pages` | Maximum number of pages to process per PDF       | All pages  |

## Summary Modes

| Mode       | Description                                                       |
|------------|-------------------------------------------------------------------|
| `brief`    | 2–3 sentence high-level summary                                   |
| `standard` | Paragraph summary with key points and main themes                 |
| `detailed` | Full breakdown: overview, key points, entities, and conclusions   |

## Output Format (JSON)
```json
{
  "file": "document.pdf",
  "pages": 12,
  "mode": "standard",
  "summary": "...",
  "key_points": ["...", "..."],
  "word_count": 4500,
  "processing_time_seconds": 3.2
}
```

## File Structure
```
pdf_summarizer_skill/
├── SKILL.md          ← This file
├── main.py           ← Entry point / CLI
├── extractor.py      ← PDF text extraction logic
├── summarizer.py     ← AI summarization logic
├── utils.py          ← Helpers (chunking, formatting, timing)
└── requirements.txt  ← Python dependencies
```
