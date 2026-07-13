"""
main.py
-------
CLI entry point for the PDF Document Summarizer Skill.

Usage:
    python main.py --pdf document.pdf
    python main.py --pdf doc1.pdf doc2.pdf --mode detailed
    python main.py --pdf document.pdf --mode brief --output results.json
    python main.py --pdf document.pdf --max-pages 10
"""

import argparse
import json
import sys
import os

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from extractor import PDFExtractor
from summarizer import PDFSummarizer
from utils import format_summary_output


def parse_args():
    parser = argparse.ArgumentParser(
        prog="pdf-summarizer",
        description="📄 PDF Document Summarizer — Extract and summarize PDF content using AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pdf report.pdf
  python main.py --pdf a.pdf b.pdf --mode detailed
  python main.py --pdf thesis.pdf --mode brief --output summary.json
  python main.py --pdf large_doc.pdf --max-pages 20
        """,
    )
    parser.add_argument(
        "--pdf",
        nargs="+",
        required=True,
        metavar="FILE",
        help="Path(s) to one or more PDF files to summarize.",
    )
    parser.add_argument(
        "--mode",
        choices=["brief", "standard", "detailed"],
        default="standard",
        help="Summary detail level: brief, standard (default), or detailed.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Save results to a JSON file (e.g. results.json).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of pages to process per PDF (default: all).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n📄 PDF Document Summarizer")
    print("=" * 60)

    # Initialize components
    extractor = PDFExtractor(ocr_fallback=True)
    try:
        summarizer = PDFSummarizer(model=args.model)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    all_results = []

    for pdf_path in args.pdf:
        print(f"\n🔍 Processing: {pdf_path}")

        # ── Step 1: Extract ────────────────────────────────────────────────
        try:
            extraction = extractor.extract(pdf_path, max_pages=args.max_pages)
        except FileNotFoundError as e:
            print(f"  ❌ {e}")
            continue
        except Exception as e:
            print(f"  ❌ Extraction failed: {e}")
            continue

        print(
            f"  ✅ Extracted {extraction['pages_processed']} / {extraction['pages']} pages"
            + (" [OCR used]" if extraction["used_ocr"] else "")
        )

        # ── Step 2: Summarize ──────────────────────────────────────────────
        try:
            result = summarizer.summarize(extraction, mode=args.mode)
        except Exception as e:
            print(f"  ❌ Summarization failed: {e}")
            continue

        # ── Step 3: Display ────────────────────────────────────────────────
        print()
        print(format_summary_output(result, args.mode))

        all_results.append(result)

    # ── Step 4: Save to JSON (optional) ───────────────────────────────────
    if args.output and all_results:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {args.output}")
        except IOError as e:
            print(f"\n⚠️  Could not save output file: {e}")

    # ── Summary stats ──────────────────────────────────────────────────────
    total = len(args.pdf)
    processed = len(all_results)
    failed = total - processed

    print(f"\n📊 Done — {processed}/{total} file(s) summarized successfully.", end="")
    if failed:
        print(f" ({failed} failed)")
    else:
        print()


if __name__ == "__main__":
    main()
