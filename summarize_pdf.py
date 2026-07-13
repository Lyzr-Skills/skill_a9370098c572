#!/usr/bin/env python3
"""
PDF Summarizer Skill
====================
Extracts text from PDF documents and generates structured summaries
using NLP techniques. Supports single files, batch processing, and
multiple detail levels.
"""

import os
import re
import sys
import math
import argparse
import datetime
from pathlib import Path
from collections import Counter

# ── PDF extraction ────────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("❌  PyMuPDF not found. Run:  pip install PyMuPDF")

# ── NLP / summarisation ───────────────────────────────────────────────────────
try:
    import nltk
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.luhn import LuhnSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
except ImportError:
    sys.exit("❌  sumy / nltk not found. Run:  pip install sumy nltk")

# ── Progress bar ──────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):          # graceful fallback
        return iterable


# ─────────────────────────────────────────────────────────────────────────────
# NLTK bootstrap
# ─────────────────────────────────────────────────────────────────────────────
def _bootstrap_nltk() -> None:
    """Download required NLTK data silently if not already present."""
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# PDF text extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> tuple[str, int]:
    """
    Extract all text from a PDF file using PyMuPDF.

    Returns
    -------
    (full_text, page_count)
    """
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    pages_text = []

    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages_text.append(text)

    doc.close()
    full_text = "\n".join(pages_text)
    return full_text, page_count


def clean_text(text: str) -> str:
    """Normalise whitespace and remove junk characters."""
    text = re.sub(r'\s+', ' ', text)               # collapse whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)    # drop non-ASCII
    text = re.sub(r'\s([?.!,;:])', r'\1', text)    # fix spacing before punct
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Keyword / keyphrase extraction (TF-based)
# ─────────────────────────────────────────────────────────────────────────────
def extract_keywords(text: str, top_n: int = 12) -> list[str]:
    """Return the top-N most significant single-word keywords."""
    stop_words = set(nltk.corpus.stopwords.words("english"))
    stop_words.update({
        "also", "would", "could", "may", "one", "two", "three",
        "however", "therefore", "thus", "since", "though", "although",
        "use", "used", "using", "new", "well", "many", "much",
    })

    tokens = nltk.word_tokenize(text.lower())
    words = [
        w for w in tokens
        if w.isalpha() and w not in stop_words and len(w) > 3
    ]

    freq = Counter(words)
    return [word for word, _ in freq.most_common(top_n)]


# ─────────────────────────────────────────────────────────────────────────────
# Core summarisation
# ─────────────────────────────────────────────────────────────────────────────
SENTENCE_COUNTS = {
    "short":    5,
    "medium":  10,
    "detailed": 20,
}

SUMMARISER_MAP = {
    "lsa":      LsaSummarizer,
    "lexrank":  LexRankSummarizer,
    "luhn":     LuhnSummarizer,
}


def summarise_text(
    text: str,
    level: str = "medium",
    language: str = "english",
    algorithm: str = "lsa",
) -> list[str]:
    """
    Run extractive summarisation and return a list of sentence strings.

    Parameters
    ----------
    text      : cleaned plain text
    level     : 'short' | 'medium' | 'detailed'
    language  : NLTK-compatible language string
    algorithm : 'lsa' | 'lexrank' | 'luhn'
    """
    n_sentences = SENTENCE_COUNTS.get(level, 10)
    SummariserClass = SUMMARISER_MAP.get(algorithm, LsaSummarizer)

    parser = PlaintextParser.from_string(text, Tokenizer(language))
    stemmer = Stemmer(language)
    summariser = SummariserClass(stemmer)
    summariser.stop_words = get_stop_words(language)

    sentences = summariser(parser.document, n_sentences)
    return [str(s) for s in sentences]


# ─────────────────────────────────────────────────────────────────────────────
# Key-point bullets (simple heuristic: first sentence of each major paragraph)
# ─────────────────────────────────────────────────────────────────────────────
def extract_key_points(text: str, max_points: int = 7) -> list[str]:
    """
    Extract bullet-point key facts by looking at paragraph-opening sentences.
    Falls back to the first N sentences if paragraphs are absent.
    """
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 80]
    points: list[str] = []

    for para in paragraphs:
        sents = nltk.sent_tokenize(para)
        if sents:
            first = sents[0].strip()
            if len(first) > 40:
                points.append(first)
        if len(points) >= max_points:
            break

    # Fallback: first N sentences
    if len(points) < 3:
        all_sents = nltk.sent_tokenize(text)
        points = [s for s in all_sents[:max_points] if len(s) > 40]

    return points[:max_points]


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────
def compute_stats(text: str, page_count: int) -> dict:
    words = text.split()
    word_count = len(words)
    sentence_count = len(nltk.sent_tokenize(text))
    avg_wpm = 200  # average adult reading speed
    reading_minutes = math.ceil(word_count / avg_wpm)

    return {
        "page_count":       page_count,
        "word_count":       word_count,
        "sentence_count":   sentence_count,
        "reading_minutes":  reading_minutes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report rendering
# ─────────────────────────────────────────────────────────────────────────────
def render_markdown(
    filename: str,
    stats: dict,
    summary_sentences: list[str],
    key_points: list[str],
    keywords: list[str],
    level: str,
) -> str:
    """Build a rich Markdown report string."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# 📄 PDF Summary Report",
        f"",
        f"> Generated on {now} · Detail level: **{level}**",
        f"",
        f"---",
        f"",
        f"## 📋 Document Info",
        f"",
        f"| Attribute        | Value                          |",
        f"|------------------|--------------------------------|",
        f"| **File**         | `{filename}`                   |",
        f"| **Pages**        | {stats['page_count']}          |",
        f"| **Words**        | {stats['word_count']:,}         |",
        f"| **Sentences**    | {stats['sentence_count']:,}     |",
        f"| **Reading Time** | ~{stats['reading_minutes']} min |",
        f"",
        f"---",
        f"",
        f"## 📝 Summary",
        f"",
    ]

    for sent in summary_sentences:
        lines.append(sent + " ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 🔑 Key Points")
    lines.append("")
    for point in key_points:
        lines.append(f"- {point}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 🏷️ Keywords")
    lines.append("")
    lines.append(", ".join(f"`{kw}`" for kw in keywords))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by PDF Summarizer Skill*")

    return "\n".join(lines)


def render_plain(
    filename: str,
    stats: dict,
    summary_sentences: list[str],
    key_points: list[str],
    keywords: list[str],
    level: str,
) -> str:
    """Build a plain-text report string."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sep = "=" * 60
    lines = [
        sep,
        "PDF SUMMARY REPORT",
        f"Generated: {now}  |  Level: {level}",
        sep,
        "",
        "DOCUMENT INFO",
        f"  File      : {filename}",
        f"  Pages     : {stats['page_count']}",
        f"  Words     : {stats['word_count']:,}",
        f"  Sentences : {stats['sentence_count']:,}",
        f"  Read Time : ~{stats['reading_minutes']} min",
        "",
        sep,
        "SUMMARY",
        sep,
        "",
        " ".join(summary_sentences),
        "",
        sep,
        "KEY POINTS",
        sep,
        "",
    ]
    for point in key_points:
        lines.append(f"  • {point}")
    lines += [
        "",
        sep,
        "KEYWORDS",
        sep,
        "",
        "  " + ", ".join(keywords),
        "",
        sep,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Single-file pipeline
# ─────────────────────────────────────────────────────────────────────────────
def process_pdf(
    pdf_path: str,
    level: str = "medium",
    fmt: str = "md",
    algorithm: str = "lsa",
) -> str:
    """
    Full pipeline: extract → clean → summarise → render.
    Returns the report string.
    """
    filename = Path(pdf_path).name

    print(f"  📖  Extracting text from '{filename}' …")
    raw_text, page_count = extract_text_from_pdf(pdf_path)

    if not raw_text.strip():
        return f"⚠️  No extractable text found in '{filename}'. The PDF may be image-based."

    text = clean_text(raw_text)

    print(f"  🔍  Analysing text ({len(text.split()):,} words) …")
    stats        = compute_stats(text, page_count)
    summary      = summarise_text(text, level=level, algorithm=algorithm)
    key_points   = extract_key_points(text)
    keywords     = extract_keywords(text)

    print(f"  ✍️   Building report …")
    if fmt == "txt":
        report = render_plain(filename, stats, summary, key_points, keywords, level)
    else:
        report = render_markdown(filename, stats, summary, key_points, keywords, level)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Batch pipeline
# ─────────────────────────────────────────────────────────────────────────────
def process_batch(
    folder: str,
    level: str = "medium",
    fmt: str = "md",
    algorithm: str = "lsa",
    output_dir: str | None = None,
) -> list[tuple[str, str]]:
    """
    Process all PDFs in a folder.
    Returns list of (pdf_path, report_string).
    """
    pdf_files = list(Path(folder).glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDF files found in '{folder}'.")
        return []

    results = []
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        print(f"\n{'─'*50}")
        print(f"📄  Processing: {pdf_path.name}")
        try:
            report = process_pdf(str(pdf_path), level=level, fmt=fmt, algorithm=algorithm)
            results.append((str(pdf_path), report))

            if output_dir:
                out_name = pdf_path.stem + ("_summary.md" if fmt == "md" else "_summary.txt")
                out_path = Path(output_dir) / out_name
                out_path.write_text(report, encoding="utf-8")
                print(f"  💾  Saved → {out_path}")
        except Exception as exc:
            print(f"  ❌  Failed: {exc}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="summarize_pdf",
        description="Summarize PDF documents using NLP — single file or batch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python summarize_pdf.py --input report.pdf
  python summarize_pdf.py --input report.pdf --level detailed --output summary.md
  python summarize_pdf.py --input ./pdfs/ --batch --output ./summaries/ --level short
  python summarize_pdf.py --input paper.pdf --format txt --algorithm lexrank
        """,
    )
    p.add_argument("--input",     required=True, help="PDF file path or folder (batch mode)")
    p.add_argument("--level",     default="medium",  choices=["short", "medium", "detailed"],
                   help="Summary detail level (default: medium)")
    p.add_argument("--output",    default=None,  help="Output file or folder path")
    p.add_argument("--batch",     action="store_true", help="Process all PDFs in the input folder")
    p.add_argument("--format",    default="md",  choices=["md", "txt"],
                   dest="fmt",   help="Output format: md or txt (default: md)")
    p.add_argument("--algorithm", default="lsa",
                   choices=["lsa", "lexrank", "luhn"],
                   help="Summarisation algorithm (default: lsa)")
    return p


def main():
    _bootstrap_nltk()

    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.batch:
        if not input_path.is_dir():
            sys.exit(f"❌  --batch requires a folder path. '{input_path}' is not a directory.")

        output_dir = args.output
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n🗂️  Batch mode — scanning '{input_path}' for PDFs …\n")
        results = process_batch(
            folder=str(input_path),
            level=args.level,
            fmt=args.fmt,
            algorithm=args.algorithm,
            output_dir=output_dir,
        )
        print(f"\n✅  Done! Processed {len(results)} PDF(s).")
        return

    # ── Single file mode ──────────────────────────────────────────────────────
    if not input_path.is_file():
        sys.exit(f"❌  File not found: '{input_path}'")
    if input_path.suffix.lower() != ".pdf":
        sys.exit(f"❌  Not a PDF file: '{input_path}'")

    print(f"\n🔎  PDF Summarizer — '{input_path.name}'\n")
    report = process_pdf(
        pdf_path=str(input_path),
        level=args.level,
        fmt=args.fmt,
        algorithm=args.algorithm,
    )

    # Output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"\n💾  Summary saved to: {out_path}")
    else:
        print(f"\n{'═'*60}")
        print(report)
        print(f"{'═'*60}\n")

    print("✅  Done!")


if __name__ == "__main__":
    main()
