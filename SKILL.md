# PDF Summarizer Skill

## Overview
This skill extracts text from PDF documents and generates structured, intelligent summaries using NLP techniques. It supports single or multiple PDFs, various summary lengths, and outputs clean markdown-formatted reports.

## Features
- EXTRACTS full text from any PDF (including multi-page documents)
- GENERATES concise, medium, or detailed summaries
- IDENTIFIES key topics, main points, and important entities
- SUPPORTS batch processing of multiple PDFs
- OUTPUTS structured markdown summary reports
- HANDLES scanned PDFs using OCR fallback (if pytesseract is available)
- PROVIDES word count, page count, and reading time statistics

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

### requirements.txt includes:
- PyMuPDF (fitz) — PDF text extraction
- sumy — NLP-based summarization
- nltk — Natural language processing
- tqdm — Progress display
- reportlab — Summary PDF export (optional)

## Usage

### Basic Usage
```bash
python summarize_pdf.py --input document.pdf
```

### Summarize with Detail Level
```bash
python summarize_pdf.py --input document.pdf --level detailed
# Levels: short | medium | detailed (default: medium)
```

### Batch Summarize Multiple PDFs
```bash
python summarize_pdf.py --input folder_of_pdfs/ --batch
```

### Save Summary to File
```bash
python summarize_pdf.py --input document.pdf --output summary.md
```

### Full Options
```bash
python summarize_pdf.py --input <file_or_folder> [--level short|medium|detailed] [--output <output_file>] [--batch] [--format md|txt]
```

## Arguments

| Argument    | Description                                      | Default   |
|-------------|--------------------------------------------------|-----------|
| `--input`   | Path to PDF file or folder (batch mode)          | Required  |
| `--level`   | Summary detail level: short, medium, detailed    | medium    |
| `--output`  | Output file path (.md or .txt)                   | stdout    |
| `--batch`   | Enable batch mode for folder of PDFs             | False     |
| `--format`  | Output format: md (markdown) or txt (plain text) | md        |

## Output Example

```
# PDF Summary Report
**File:** research_paper.pdf
**Pages:** 12 | **Words:** 4,823 | **Est. Reading Time:** ~19 min

## Summary
This paper explores the impact of transformer models on NLP benchmarks...

## Key Points
- Transformer architectures outperform RNNs in translation tasks
- Attention mechanisms allow better long-range dependency modeling
- BERT achieves state-of-the-art results on GLUE benchmark

## Keywords
transformers, NLP, BERT, attention, deep learning, benchmarks
```

## Notes
- Works best with text-based PDFs (not image-only scans)
- For scanned PDFs, install `pytesseract` and `Pillow` for OCR support
- Large PDFs (100+ pages) may take a few seconds to process
