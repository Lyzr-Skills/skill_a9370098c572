"""
summarizer.py
-------------
AI-powered summarization engine using OpenAI GPT models.
Supports three modes: brief, standard, and detailed.
Handles large documents via chunked summarization (map-reduce pattern).
"""

import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from utils import chunk_text, count_words, timer

load_dotenv()


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert document analyst and summarizer. "
    "Your task is to produce clear, accurate, and well-structured summaries of documents. "
    "Always be concise yet comprehensive. Avoid filler phrases."
)

MODE_PROMPTS = {
    "brief": (
        "Provide a BRIEF summary of the following document text in 2–3 sentences. "
        "Capture only the absolute essence.\n\nDocument Text:\n{text}"
    ),
    "standard": (
        "Provide a STANDARD summary of the following document text. Include:\n"
        "1. A clear overview paragraph (3–5 sentences).\n"
        "2. A bullet list of 4–6 key points.\n\n"
        "Format your response as:\n"
        "OVERVIEW:\n<overview paragraph>\n\n"
        "KEY POINTS:\n- <point 1>\n- <point 2>\n...\n\n"
        "Document Text:\n{text}"
    ),
    "detailed": (
        "Provide a DETAILED summary of the following document text. Include:\n"
        "1. A comprehensive overview (5–8 sentences).\n"
        "2. A bullet list of 6–10 key points.\n"
        "3. Notable entities mentioned (people, organizations, dates, locations) — list them.\n"
        "4. A concluding statement about the document's significance or takeaways.\n\n"
        "Format your response as:\n"
        "OVERVIEW:\n<overview>\n\n"
        "KEY POINTS:\n- <point>\n...\n\n"
        "ENTITIES:\nPeople: ...\nOrganizations: ...\nDates: ...\nLocations: ...\n\n"
        "CONCLUSION:\n<conclusion>\n\n"
        "Document Text:\n{text}"
    ),
}

CHUNK_COMBINE_PROMPT = (
    "The following are partial summaries from different sections of a large document. "
    "Combine them into a single coherent {mode} summary.\n\n"
    "Partial Summaries:\n{text}"
)


class PDFSummarizer:
    """
    Summarizes text extracted from PDFs using OpenAI GPT models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_chunk_tokens: int = 3000,
    ):
        """
        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: OpenAI model to use.
            max_chunk_tokens: Max tokens per text chunk sent to the model.
        """
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in your environment or .env file."
            )
        self.client = OpenAI(api_key=key)
        self.model = model
        self.max_chunk_tokens = max_chunk_tokens

    @timer
    def summarize(self, extraction_result: dict, mode: str = "standard") -> dict:
        """
        Generate a summary from an extraction result dict.

        Args:
            extraction_result: Dict returned by PDFExtractor.extract().
            mode: One of 'brief', 'standard', 'detailed'.

        Returns:
            Summary result dict with keys: summary, key_points, entities,
            conclusion, word_count, pages, pages_processed, used_ocr, file, title, author.
        """
        if mode not in MODE_PROMPTS:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {list(MODE_PROMPTS.keys())}")

        text = extraction_result["text"]
        word_count = count_words(text)

        if not text.strip():
            return self._empty_result(extraction_result, mode)

        # Chunk large documents
        chunks = chunk_text(text, max_tokens=self.max_chunk_tokens)

        if len(chunks) == 1:
            raw_summary = self._call_model(MODE_PROMPTS[mode].format(text=chunks[0]))
        else:
            # Map step: summarize each chunk briefly
            partial_summaries = []
            for i, chunk in enumerate(chunks):
                print(f"  → Summarizing chunk {i + 1}/{len(chunks)}...")
                partial = self._call_model(MODE_PROMPTS["brief"].format(text=chunk))
                partial_summaries.append(f"Section {i + 1}: {partial}")

            # Reduce step: combine partial summaries
            combined_text = "\n\n".join(partial_summaries)
            reduce_prompt = CHUNK_COMBINE_PROMPT.format(mode=mode, text=combined_text)
            raw_summary = self._call_model(reduce_prompt)

        # Parse structured output
        parsed = self._parse_response(raw_summary, mode)

        return {
            "file": extraction_result.get("file_name", "unknown.pdf"),
            "file_path": extraction_result.get("file_path", ""),
            "title": extraction_result.get("title", ""),
            "author": extraction_result.get("author", ""),
            "pages": extraction_result.get("pages", 0),
            "pages_processed": extraction_result.get("pages_processed", 0),
            "used_ocr": extraction_result.get("used_ocr", False),
            "mode": mode,
            "word_count": word_count,
            "summary": parsed.get("summary", raw_summary),
            "key_points": parsed.get("key_points", []),
            "entities": parsed.get("entities", {}),
            "conclusion": parsed.get("conclusion", ""),
        }

    def _call_model(self, user_prompt: str) -> str:
        """
        Send a prompt to the OpenAI model and return the text response.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _parse_response(self, raw: str, mode: str) -> dict:
        """
        Parse structured model output into a clean dict.
        """
        result = {"summary": raw, "key_points": [], "entities": {}, "conclusion": ""}

        lines = raw.splitlines()
        section = None
        overview_lines = []
        key_point_lines = []
        entity_lines = []
        conclusion_lines = []

        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("OVERVIEW:"):
                section = "overview"
                rest = stripped[len("OVERVIEW:"):].strip()
                if rest:
                    overview_lines.append(rest)
            elif upper.startswith("KEY POINTS:"):
                section = "key_points"
            elif upper.startswith("ENTITIES:"):
                section = "entities"
            elif upper.startswith("CONCLUSION:"):
                section = "conclusion"
                rest = stripped[len("CONCLUSION:"):].strip()
                if rest:
                    conclusion_lines.append(rest)
            else:
                if section == "overview":
                    overview_lines.append(stripped)
                elif section == "key_points":
                    if stripped.startswith(("-", "•", "*")):
                        key_point_lines.append(stripped.lstrip("-•* ").strip())
                    elif stripped:
                        key_point_lines.append(stripped)
                elif section == "entities":
                    if ":" in stripped:
                        entity_lines.append(stripped)
                elif section == "conclusion":
                    if stripped:
                        conclusion_lines.append(stripped)

        if overview_lines:
            result["summary"] = " ".join(overview_lines).strip()
        if key_point_lines:
            result["key_points"] = [p for p in key_point_lines if p]
        if entity_lines:
            for ent_line in entity_lines:
                parts = ent_line.split(":", 1)
                if len(parts) == 2:
                    ent_type = parts[0].strip()
                    ent_values = [v.strip() for v in parts[1].split(",") if v.strip() and v.strip().lower() != "none"]
                    if ent_values:
                        result["entities"][ent_type] = ent_values
        if conclusion_lines:
            result["conclusion"] = " ".join(conclusion_lines).strip()

        return result

    def _empty_result(self, extraction_result: dict, mode: str) -> dict:
        """Return a result dict for PDFs with no extractable text."""
        return {
            "file": extraction_result.get("file_name", "unknown.pdf"),
            "file_path": extraction_result.get("file_path", ""),
            "title": extraction_result.get("title", ""),
            "author": extraction_result.get("author", ""),
            "pages": extraction_result.get("pages", 0),
            "pages_processed": extraction_result.get("pages_processed", 0),
            "used_ocr": extraction_result.get("used_ocr", False),
            "mode": mode,
            "word_count": 0,
            "summary": "No extractable text found in this PDF.",
            "key_points": [],
            "entities": {},
            "conclusion": "",
            "processing_time_seconds": 0,
        }
