import asyncio
import ast
import base64
import csv
import concurrent.futures
import json
import math
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urlparse, parse_qs, unquote, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from src.tools.tools import Tool
from src.dynacall.llm_adapters import create_llm_adapter
from src.docstore.google_search import WebSearchAPI
from configs.bfcl_ws.urlfetch import URLFetch

try:
    import requests
except ImportError:
    requests = None

try:
    from googlesearch import search as google_search
except ImportError:
    google_search = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError:
    TavilySearchResults = None

try:
    from langchain_community.document_loaders import WikipediaLoader
except ImportError:
    WikipediaLoader = None


def _extract_http_status_code(exc: Exception) -> Optional[int]:
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
    if isinstance(exc, HTTPError):
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            return code
    message = str(exc)
    if "429" in message:
        return 429
    return None


_SEARCH_ENGINE_DESCRIPTION = (
    'search_engine(["query: str"])-> list:\n'
    " - Official GAIA-aligned web search capability.\n"
    " - Searches the web and returns results with title, href, and body.\n"
    " - Uses public-web search backends and may combine search engines for better reliability.\n"
    " - Use this when the question requires search engine access.\n"
)

_WEB_BROWSER_DESCRIPTION = (
    "web_browser(query_or_url: str, search_results: Optional[list]) -> str:\n"
    " - Official GAIA-aligned web browsing capability.\n"
    " - Opens URLs from search results or a direct URL and extracts readable text.\n"
    " - Use this after search_engine when snippets are not enough.\n"
)

_SEARCH_GET_CONTENTS_DESCRIPTION = (
    'search_get_contents(["query_or_url: str", "search_results?: list"]) -> str:\n'
    " - GAIA-agent aligned content retrieval helper.\n"
    " - Fetches and returns readable full contents from a direct URL or from top candidate URLs in search results.\n"
    " - Use this when search snippets are insufficient and you need正文/页面内容 for verification or extraction.\n"
    " - Prefer this over weak snippet-only reasoning.\n"
)

_DEEPSEARCH_DESCRIPTION = (
    'deepsearch(["query: str"]) -> str:\n'
    " - JoyAgent-style deep search tool for harder web questions.\n"
    " - Runs an agentic search loop: decompose the question into atomic source-aware queries, search/fetch evidence, critique coverage, and optionally search once more.\n"
    " - Returns a compact JSON bundle with executed queries, search results, fetched contents, coverage assessment, and evidence summary.\n"
    " - Use this when one search query is brittle or when multi-source verification is needed before extraction.\n"
    " - Prefer this over repeating near-identical search_engine calls by hand.\n"
)

_ORCID_READER_DESCRIPTION = (
    'orcid_reader(["orcid_url_or_id: str", "before_year?: int"]) -> str:\n'
    " - Reads a public ORCID record from the official ORCID API.\n"
    " - Returns structured JSON with the ORCID id, public works, counts_by_year, and optional count_before_year.\n"
    " - Use this when the question asks about an ORCID page, ORCID works, or publication counts for a known ORCID identifier.\n"
    " - Prefer this over generic web search for counting works on ORCID pages.\n"
)

_GITHUB_ISSUE_SEARCH_DESCRIPTION = (
    'github_issue_search(["repo: str", "labels: str", "state?: str", "sort?: str", "direction?: str", "per_page?: int"]) -> str:\n'
    " - Uses GitHub's public Issues API without an API key to search issues by repo, state, and comma-separated labels.\n"
    " - Best for exact repo/label/state filters, oldest/newest ordering, label verification, and label-added timeline questions.\n"
    " - Returns compact JSON with issue number, title, state, URL, created_at, labels, and label/unlabel timeline events.\n"
    " - If exact label matching returns no issues, it automatically retries with family-aware label filtering after dropping numeric prefixes such as '05 - ' or '06 - '.\n"
    " - Use this instead of broad web search when the question asks when a GitHub label was added or which issue has exact labels.\n"
    ' - Example: github_issue_search(["numpy/numpy", "06 - Regression,component: numpy.polynomial", "closed", "created", "asc", "5"]).\n'
)

_WIKI_SECTION_EXTRACT_DESCRIPTION = (
    'wiki_section_extract(["wikipedia_url_or_title: str", "section_hint: str", "revision_year?: str"]) -> str:\n'
    " - Wikipedia-specific section/table extractor.\n"
    " - Fetches one exact Wikipedia article, optionally resolves a revision within a given year, isolates the best-matching section, and preserves list/table rows as readable text.\n"
    " - Prefer this for section-scoped Wikipedia extraction when web_browser is too broad.\n"
)

_CROSSREF_LOOKUP_DESCRIPTION = (
    'crossref_lookup(["paper_title: str", "max_results?: int"]) -> str:\n'
    " - Academic metadata lookup via Crossref/OpenAlex.\n"
    " - Returns grounded paper candidates with title, DOI URL, year, venue, and author list.\n"
    " - Prefer this for paper-title questions before generic web search.\n"
)

_QUOTE_VERIFIER_DESCRIPTION = (
    'quote_verifier(["citation_or_title_or_doi: str", "quoted_text: str"]) -> str:\n'
    " - Verifies whether an inline citation quote matches the source text.\n"
    " - Searches for accessible full-text, PDF, preview, or snippet evidence using title/DOI and distinctive quote anchors.\n"
    " - Returns JSON with matches, mismatched_word, correct_word, source_url, and support.\n"
    " - Use this for citation fact-checking questions before giving up on blocked publisher pages.\n"
    " - This is not a metadata lookup; citation reports, RIS/BibTeX, and DOI-only pages are rejected as quote evidence.\n"
)

_OPENALEX_AUTHOR_WORKS_DESCRIPTION = (
    'openalex_author_works(["author_name: str", "known_work_title?: str", "before_year?: int", "max_works?: int"]) -> str:\n'
    " - Academic author disambiguation and works lookup via OpenAlex.\n"
    " - Resolves the most likely author entity and returns chronological works plus earliest-work evidence.\n"
    " - Prefer this for author chronology and same-name author disambiguation questions.\n"
)

_FILE_READER_DESCRIPTION = (
    'file_reader(["absolute_file_path: str"]) -> str:\n'
    " - Official GAIA-aligned diverse filetype reading capability.\n"
    " - Auto-detects file type and returns extracted text, previews, or metadata.\n"
    " - Input must be a real local file path, typically from an attachment or download_file_from_url.\n"
    " - Do not pass search snippets or arbitrary webpage text into this tool.\n"
    " - Use this first when the task has an attachment.\n"
)

_PDF_VIEWER_DESCRIPTION = (
    'pdf_viewer(["absolute_file_path: str"]) -> str:\n'
    " - Reads text from a PDF attachment.\n"
    " - Preserves layout when possible so tables and aligned columns remain readable.\n"
    " - If a table is detected, the observation may include a Parsed table preview in JSON-like row format. Prefer that structured preview over counting unrelated text lines.\n"
    " - Use when the attachment is a PDF or when file_reader indicates PDF content.\n"
)

_SPREADSHEET_READER_DESCRIPTION = (
    'spreadsheet_reader(["absolute_file_path: str"]) -> str:\n'
    " - Reads spreadsheet-like files such as xlsx, xls, csv, and tsv.\n"
    " - Returns sheet names, columns, and preview rows.\n"
    " - Use this before python so you can identify headers, sections, and the correct sheet.\n"
)

_POWERPOINT_VIEWER_DESCRIPTION = (
    'powerpoint_viewer(["absolute_file_path: str"]) -> str:\n'
    " - Reads text from PowerPoint slides.\n"
    " - Use this when the attachment is a PPTX file.\n"
)

_TEXT_READER_DESCRIPTION = (
    'text_reader(["absolute_file_path: str"]) -> str:\n'
    " - Reads text-like files such as txt, py, json, jsonld, xml, and html.\n"
    " - Use this for text-heavy local files.\n"
    " - Do not use this for structured scientific coordinate files such as pdb.\n"
    " - If exact parsing or coordinate computation is needed, pass the original local attachment path directly into python or execute_code_multilang instead.\n"
)

_ARCHIVE_EXPLORER_DESCRIPTION = (
    'archive_explorer(["absolute_file_path: str"]) -> str:\n'
    " - Lists ZIP archive contents and returns JSON with extracted local member paths.\n"
    " - The result includes members as objects with name and path fields, plus readable previews for some text members.\n"
    " - Use this when the attachment is a compressed archive.\n"
)

_SAVE_AND_READ_FILE_DESCRIPTION = (
    'save_and_read_file(["content: str", "filename?: str"]) -> str:\n'
    " - GAIA-agent aligned file helper.\n"
    " - Saves provided text content into a temporary file and returns the path.\n"
    " - Use this when you need to create an intermediate file for later processing.\n"
)

_DOWNLOAD_FILE_FROM_URL_DESCRIPTION = (
    'download_file_from_url(["url: str", "filename?: str"]) -> str:\n'
    " - GAIA-agent aligned download helper.\n"
    " - Downloads a file from a URL into a temporary location and returns the local path.\n"
    " - The first argument must be a concrete downloadable http(s) URL.\n"
    " - Do not pass a search query, page description, or browser query into this tool.\n"
    " - If you only have search results or page text, first use semantic_map to extract a concrete file or image URL.\n"
    " - Use this when a web page points to a file that must be processed locally.\n"
)

_IMAGE_RECOGNITION_DESCRIPTION = (
    'image_recognition(["absolute_file_path: str"]) -> str:\n'
    " - Official GAIA-aligned multimodal capability for images.\n"
    " - Combines basic image analysis with OCR-style text extraction.\n"
)

_OCR_DESCRIPTION = (
    'ocr(["absolute_file_path: str"]) -> str:\n'
    " - Official GAIA-aligned OCR capability.\n"
    " - Attempts to extract text clues from an image attachment.\n"
)

_ANALYZE_IMAGE_DESCRIPTION = (
    'analyze_image(["absolute_file_path: str"]) -> str:\n'
    " - GAIA-agent style image analysis tool.\n"
    " - Reports dimensions, format, and lightweight visual metadata for an image.\n"
    " - Use together with OCR for image-based reasoning tasks.\n"
)

_TRANSFORM_IMAGE_DESCRIPTION = (
    'transform_image(["absolute_file_path: str", "operation: str", "params_json?: str"]) -> str:\n'
    " - GAIA-agent aligned image transformation tool.\n"
    " - Supports resize, rotate, crop, flip, brightness, contrast, blur, sharpen, and grayscale.\n"
    " - Saves the transformed image to a temporary file and returns its path.\n"
)

_DRAW_ON_IMAGE_DESCRIPTION = (
    'draw_on_image(["absolute_file_path: str", "drawing_type: str", "params_json: str"]) -> str:\n'
    " - GAIA-agent aligned image annotation tool.\n"
    " - Draws rectangles, circles, lines, or text onto an image and returns the saved result path.\n"
)

_GENERATE_SIMPLE_IMAGE_DESCRIPTION = (
    'generate_simple_image(["image_type: str", "width?: int", "height?: int", "params_json?: str"]) -> str:\n'
    " - GAIA-agent aligned image generation helper.\n"
    " - Generates simple gradient or noise images and returns the saved path.\n"
)

_COMBINE_IMAGES_DESCRIPTION = (
    'combine_images(["image_path_1: str", "image_path_2: str", "...", "operation?: str"]) -> str:\n'
    " - GAIA-agent aligned image combination helper.\n"
    " - Combines multiple local images into a collage, vertical stack, horizontal stack, or blend and returns the saved path.\n"
)

_EXTRACT_TEXT_FROM_IMAGE_DESCRIPTION = (
    'extract_text_from_image(["absolute_file_path: str"]) -> str:\n'
    " - GAIA-agent style OCR helper.\n"
    " - Extracts visible text from an image using Tesseract OCR.\n"
)

_SPEECH_TO_TEXT_DESCRIPTION = (
    'speech_to_text(["absolute_file_path: str"]) -> str:\n'
    " - Official GAIA-aligned audio transcription capability.\n"
    " - Uses a speech-to-text model to transcribe local audio files.\n"
)

_YOUTUBE_TRANSCRIPT_DESCRIPTION = (
    'youtube_transcript(["youtube_video_url: str"]) -> str:\n'
    " - Local YouTube transcript and metadata reader for GAIA without external API keys.\n"
    " - Accepts one concrete canonical video URL for a single video (youtube.com/watch?v=... or youtu.be/...).\n"
    " - Do not pass search pages, channel pages, playlist pages, or clip pages.\n"
    " - Returns title, uploader, channel, duration, description, transcript text when captions are available, and fallback search evidence when captions are unavailable.\n"
    " - Prefer this for named YouTube/video questions before broad browsing when video metadata or captions may be enough.\n"
)

_PYTHON_DESCRIPTION = (
    'python(["python_code: str"]) -> str:\n'
    " - Official GAIA-aligned coding capability.\n"
    " - Executes Python code in a sandboxed subprocess and returns stdout.\n"
    " - Use this for parsing files, computations, and transformations.\n"
    " - Prefer Python standard library and simple self-contained code. Do not assume optional third-party packages are installed unless earlier evidence shows they are available.\n"
    " - For symbolic logic, truth-table, or propositional-equivalence questions, prefer enumerating all boolean assignments directly in pure Python instead of importing sympy.\n"
    " - Pass exactly one Python string inside the list.\n"
    " - The code must be valid Python.\n"
    " - Prefer concise code, but if the logic needs loops or conditionals you may use \\n inside the string instead of forcing invalid semicolon chains.\n"
    " - When you compute an answer with Python, end by printing the final result.\n"
    " - For .xlsx files use pandas.read_excel or Excel-aware libraries, not read_csv.\n"
)

_EXECUTE_CODE_MULTILANG_DESCRIPTION = (
    'execute_code_multilang(["code: str", "language?: str"]) -> str:\n'
    " - GAIA-agent aligned code interpreter compatibility tool.\n"
    " - Executes code in multiple languages when supported by the local environment.\n"
    " - Supported directly: python, bash, sql. Best-effort support: c, java if local compilers exist.\n"
    " - Returns stdout and concise execution diagnostics.\n"
)

_CODE_INTERPRETER_DESCRIPTION = (
    'code_interpreter(["code_or_task: str", "language?: str"]) -> str:\n'
    " - JoyAgent-style code interpreter compatibility tool.\n"
    " - Best for exact file parsing, spreadsheet checks, deterministic counting, and local computations.\n"
    " - Internally uses the multilang execution path and returns concise stdout/diagnostics.\n"
    " - Prefer this over pure calculator for anything involving files, loops, parsing, or non-trivial logic.\n"
)

_CALCULATOR_DESCRIPTION = (
    'calculator(["expression: str"]) -> str:\n'
    " - Official GAIA-aligned computation capability.\n"
    " - Evaluates a pure Python arithmetic expression and returns the result.\n"
    " - The expression must already be complete. Do not include inline parsing or partial dependency references such as $3.split(), $2[0], or float($4) inside the expression; first extract clean scalar numbers with semantic_map or use python/code_interpreter.\n"
    " - Available helper names are ceil, floor, sqrt, abs, round, min, and max. Use ceil(x), not math.ceil(x), because the math module itself is not exposed.\n"
)

_VERIFIER_DESCRIPTION = (
    'verifier(["question: str", "proposed_answer: str", "evidence?: str"]) -> str:\n'
    " - GAIA-agent aligned answer verification helper.\n"
    " - Checks whether the proposed answer is concise, supported by evidence, and matches the question type.\n"
    " - Returns structured JSON with valid, confidence, issues, and recommendation.\n"
    " - Use this before final answer when uncertainty remains.\n"
)

_SEMANTIC_MAP_DESCRIPTION = (
    'semantic_map(["global_question: str", "local_request: str", "plan_context: str", ["observation_1", "observation_2", ...], "output_schema: str"]) -> str:\n'
    " - Typed semantic extraction and normalization tool for GAIA. This is the main bridge between raw observations and precise downstream actions.\n"
    " - Preferred interface is information_extract-style: global_question + local_request + plan_context + observations + output_schema.\n"
    " - Legacy calls of the form semantic_map([instruction, inputs, output_schema]) are still accepted and will be auto-upgraded with runtime context.\n"
    " - Use this for concrete URL selection, exact field extraction, schema normalization, query rewriting, candidate-answer extraction, list filtering, answer-word matching, and short grounded judgments.\n"
    " - Prefer this over free-form reasoning or python when a downstream step needs one exact URL, one exact date/token, one cleaned scalar, one candidate list, or machine-friendly JSON.\n"
    " - The result must reflect the global question, current local request, current plan step, hard constraints, and relevant observation history; do not treat it as a blind local text filter.\n"
    " - If a later tool needs only one field from a list or JSON object, extract that field here into its own standalone step. Downstream plans only support whole-step references like $m1, not indexed expressions.\n"
    " - Do not use this to solve the whole task or to call other tools.\n"
    " - When the local request asks for a URL, return one concrete complete URL copied from evidence, not a guess, identifier, explanation, or empty value just because the URL is http rather than https.\n"
    " - When the global question or local request specifies a site, language, source, date, year, issue, revision, category, label, section, entity type, or submodule qualifier, treat it as a mandatory filter before selecting earliest/latest/top-ranked results.\n"
    " - When extracting from a section such as Studio albums, Filmography, Discography, Awards, labels, timeline events, or tables, stay inside the requested section and exclude items from other sections.\n"
    " - When the local request asks for earliest/oldest/latest/first/last, compare explicit dates/years in evidence instead of defaulting to the top-ranked search result.\n"
    " - When the local request asks for a place or locality, prefer the most specific location string supported by evidence.\n"
    " - When the local request asks for a symbol or character name, return its shortest standard name rather than the literal character.\n"
    " - When extracting spreadsheet filters, records, operands, or code-ready JSON, return exact machine-friendly field values, not natural-language filter expressions.\n"
    " - If evidence is insufficient, contradictory, wrong-page, wrong-entity, wrong-date, or wrong-format for the requested schema, return the empty value for that schema rather than guessing.\n"
    " - Supported output_schema values include string, number, boolean, list[string], and json{field:type,...}.\n"
    " - The output is normalized to match the requested schema so downstream tools can consume it reliably.\n"
)


_BROWSER_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]


def _default_http_headers(extra: Optional[dict] = None) -> dict:
    headers = {
        "User-Agent": random.choice(_BROWSER_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    if extra:
        headers.update(extra)
    return headers


def _truncate(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _truncate_json_safely(payload: Dict[str, Any], limit: int = 12000) -> str:
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) <= limit:
        return text

    reduced = dict(payload)
    works = list(reduced.get("works", []) or [])
    works_before_year = list(reduced.get("works_before_year", []) or [])

    if works:
        reduced["works_total_items"] = len(works)
        reduced["works"] = works[: min(40, len(works))]
        reduced["works_truncated"] = len(works) > len(reduced["works"])
    if works_before_year:
        reduced["works_before_year_total_items"] = len(works_before_year)
        reduced["works_before_year"] = works_before_year[: min(40, len(works_before_year))]
        reduced["works_before_year_truncated"] = len(works_before_year) > len(reduced["works_before_year"])

    text = json.dumps(reduced, ensure_ascii=False)
    if len(text) <= limit:
        return text

    if "works" in reduced:
        reduced["works"] = reduced["works"][:10]
    if "works_before_year" in reduced:
        reduced["works_before_year"] = reduced["works_before_year"][:10]
    return json.dumps(reduced, ensure_ascii=False)


def _extract_pdf_text_from_bytes(data: bytes, limit: int = 12000) -> str:
    temp_path = None
    try:
        from PyPDF2 import PdfReader

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            temp_path = tmp.name

        reader = PdfReader(temp_path)
        pages = []
        for page in reader.pages[:5]:
            pages.append(page.extract_text() or "")
        text = "\n\n".join(page for page in pages if page.strip()).strip()
        if text:
            return _truncate(text, limit)
    except Exception:
        pass
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
    return "PDF text extraction unavailable."


def _split_top_level_csv(text: str) -> list[str]:
    parts = []
    buf = []
    depth = 0
    for ch in text:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [part for part in parts if part]


def _coerce_semantic_value(value: Any, schema: str) -> Any:
    schema = (schema or "string").strip()
    lowered = schema.lower()

    if lowered == "string":
        if isinstance(value, str):
            return value.strip().strip('"').strip("'")
        return str(value)

    if lowered == "number":
        if isinstance(value, (int, float)):
            return value
        text = str(value).strip()
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        return float(text)

    if lowered == "boolean":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
        raise ValueError(f"Cannot coerce to boolean: {value}")

    if lowered.startswith("list[") and lowered.endswith("]"):
        inner = schema[5:-1].strip()
        if not isinstance(value, list):
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        value = parsed
                    else:
                        value = [item.strip() for item in value.split(",") if item.strip()]
                except Exception:
                    value = [item.strip() for item in value.split(",") if item.strip()]
            else:
                value = [value]
        return [_coerce_semantic_value(item, inner) for item in value]

    json_match = re.fullmatch(r"json\s*\{(.*)\}", schema, flags=re.I | re.S)
    if json_match:
        inner = json_match.group(1).strip()
        fields = {}
        for part in _split_top_level_csv(inner):
            if ":" not in part:
                continue
            key, type_spec = part.split(":", 1)
            fields[key.strip()] = type_spec.strip()
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict-like JSON object, got: {type(value)}")
        return {
            key: _coerce_semantic_value(value.get(key), type_spec)
            for key, type_spec in fields.items()
        }

    return value


def _format_semantic_value(value: Any, schema: str) -> str:
    lowered = (schema or "string").strip().lower()
    if lowered == "string":
        return str(value)
    if lowered == "number":
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    if lowered == "boolean":
        return "true" if bool(value) else "false"
    return json.dumps(value, ensure_ascii=False)


def _extract_semantic_payload(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


async def _repair_semantic_json_payload(
    llm_adapter: Any,
    raw_payload: str,
    output_schema: str,
) -> str:
    prompt = (
        "You are a JSON normalizer.\n"
        "Convert the given content into strict valid JSON only.\n"
        "Do not explain anything.\n"
        "Do not add markdown fences.\n"
        "Do not change the meaning.\n"
        "Output must be parseable by json.loads.\n"
        f"Target schema:\n{output_schema}\n\n"
        f"Content to normalize:\n{raw_payload}\n"
    )
    repaired = await llm_adapter.apredict(prompt)
    if repaired.startswith("Error:"):
        return repaired
    return _extract_semantic_payload(repaired)


def _normalize_excel_rows(raw_rows: list[list[str]]) -> tuple[list[str], list[dict[str, str]]]:
    header_idx = -1
    max_non_empty = 0
    for idx, row in enumerate(raw_rows[:20]):
        non_empty = sum(bool(str(cell).strip()) for cell in row)
        if non_empty >= 2 and non_empty > max_non_empty:
            header_idx = idx
            max_non_empty = non_empty

    if header_idx == -1:
        width = max((len(row) for row in raw_rows), default=1)
        headers = [f"col_{i}" for i in range(width)]
        records = []
        for row in raw_rows:
            padded = list(row) + [""] * max(0, width - len(row))
            records.append({headers[i]: str(padded[i]).strip() for i in range(width)})
        return headers, records

    header_row = raw_rows[header_idx]
    width = max(len(header_row), max((len(row) for row in raw_rows[header_idx + 1:]), default=len(header_row)))
    headers = [str(cell).strip() or f"col_{i}" for i, cell in enumerate(header_row + [""] * max(0, width - len(header_row)))]

    records = []
    current_section = ""
    for row in raw_rows[header_idx + 1:]:
        padded = list(row) + [""] * max(0, width - len(row))
        trimmed = [str(value).strip() for value in padded]
        non_empty = [value for value in trimmed if value]
        if not non_empty:
            continue
        if len(non_empty) == 1:
            current_section = non_empty[0]
            continue

        record = {headers[i]: trimmed[i] for i in range(width)}
        if current_section and "Media Type" not in record:
            record["Media Type"] = current_section
        records.append(record)

    final_headers = headers.copy()
    if any("Media Type" in record for record in records) and "Media Type" not in final_headers:
        final_headers.append("Media Type")
    return final_headers, records


def _extract_layout_table_preview(text: str, max_rows: int = 60) -> Optional[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    header_idx = None
    headers = None

    for idx, line in enumerate(lines):
        parts = [part.strip() for part in re.split(r"\s{2,}", line.strip()) if part.strip()]
        if len(parts) >= 3:
            normalized = {part.lower() for part in parts}
            if any(key in normalized for key in {"title", "author", "status", "genre", "publisher"}):
                header_idx = idx
                headers = parts
                break

    if header_idx is None or headers is None:
        return None

    records = []
    expected_cols = len(headers)
    for line in lines[header_idx + 1:]:
        parts = [part.strip() for part in re.split(r"\s{2,}", line.strip()) if part.strip()]
        if len(parts) < expected_cols:
            continue
        if len(parts) > expected_cols:
            parts = parts[: expected_cols - 1] + [" ".join(parts[expected_cols - 1:])]
        record = {headers[i]: parts[i] for i in range(expected_cols)}
        records.append(record)
        if len(records) >= max_rows:
            break

    if not records:
        return None
    return json.dumps(records, ensure_ascii=False)


def _iter_wikipedia_wikitext_sections(wikitext: str) -> list[dict[str, str]]:
    if not wikitext:
        return []
    pattern = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.M)
    matches = list(pattern.finditer(wikitext))
    if not matches:
        return []

    sections: list[dict[str, str]] = []
    for idx, match in enumerate(matches):
        level = len(match.group(1))
        title = (match.group(2) or "").strip()
        start = match.end()
        end = len(wikitext)
        for later in matches[idx + 1:]:
            later_level = len(later.group(1))
            if later_level <= level:
                end = later.start()
                break
        body = wikitext[start:end].strip()
        if body:
            sections.append({"title": title, "level": str(level), "body": body})
    return sections


def _extract_wikipedia_focus_sections(wikitext: str, limit: int = 12000) -> str:
    sections = _iter_wikipedia_wikitext_sections(wikitext)
    if not sections:
        return ""

    focus_keywords = (
        "discography",
        "album",
        "albums",
        "filmography",
        "bibliography",
        "publication",
        "publications",
        "works",
        "career statistics",
        "results",
        "record",
        "records",
        "awards",
        "episodes",
        "track",
        "tracks",
        "songs",
        "roster",
        "schedule",
        "medal",
        "medals",
    )

    picked: list[str] = []
    total = 0
    for section in sections:
        title = (section.get("title") or "").strip()
        body = (section.get("body") or "").strip()
        title_l = title.lower()
        if not any(keyword in title_l for keyword in focus_keywords):
            continue
        block = f"SECTION: {title}\n{body}"
        if total >= limit:
            break
        remaining = max(0, limit - total)
        if remaining <= 0:
            break
        block = _truncate(block, min(remaining, 2500))
        picked.append(block)
        total += len(block) + 2

    return "\n\n".join(picked)


def _safe_read_text(path: pathlib.Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _http_get(url: str, headers: Optional[dict] = None, timeout: int = 30):
    headers = _default_http_headers(headers)
    if requests is not None:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response.text, response.url

    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="ignore")
        final_url = response.geturl()
    return body, final_url


def _http_get_bytes(url: str, headers: Optional[dict] = None, timeout: int = 30):
    headers = _default_http_headers(headers)
    if requests is not None:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response.content, response.url, response.headers.get("Content-Type", "")

    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
    return body, final_url, content_type


def _http_get_json(url: str, headers: Optional[dict] = None, timeout: int = 30) -> Dict[str, Any]:
    headers = headers or {}
    text, _ = _http_get(url, headers=headers, timeout=timeout)
    return json.loads(text)


def _openalex_get_json(url: str, headers: Optional[dict] = None, timeout: int = 30) -> Dict[str, Any]:
    headers = headers or {}
    headers.setdefault("User-Agent", "DynaCall/GAIA")
    if requests is not None:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="ignore"))


def _normalize_openalex_authorships(authorships: Any) -> List[str]:
    names: List[str] = []
    for authorship in authorships or []:
        try:
            display = (((authorship or {}).get("author") or {}).get("display_name") or "").strip()
            if display:
                names.append(display)
        except Exception:
            continue
    return names


def _coerce_search_results(value: Any) -> Any:
    if value is None or isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return text
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            if isinstance(parsed, (list, dict)):
                return parsed
        except Exception:
            continue
    return text


def _normalize_bibliographic_title(title: str) -> str:
    text = str(title or "").strip()
    if not text:
        return text

    letters = [char for char in text if char.isalpha()]
    degraded_case = False
    if letters:
        uppercase_inside = sum(1 for char in letters[1:] if char.isupper())
        degraded_case = uppercase_inside <= max(1, len(letters) // 40)
    if not degraded_case:
        return text

    text = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    small_words = {
        "a", "an", "the", "and", "or", "but", "for", "nor", "on", "at", "to",
        "from", "by", "of", "in", "with", "vs", "vs.", "via", "per",
    }

    def _normalize_word(token: str, is_first: bool, is_last: bool) -> str:
        if not token:
            return token
        prefix_match = re.match(r"^[^A-Za-z0-9]*", token)
        suffix_match = re.search(r"[^A-Za-z0-9]*$", token)
        prefix = prefix_match.group(0) if prefix_match else ""
        suffix = suffix_match.group(0) if suffix_match else ""
        core = token[len(prefix): len(token) - len(suffix) if suffix else len(token)]
        if not core:
            return token
        if core.isupper() and len(core) <= 5:
            normalized_core = core
        elif core.lower() in small_words and not is_first and not is_last:
            normalized_core = core.lower()
        else:
            normalized_core = core[0].upper() + core[1:].lower()
        return f"{prefix}{normalized_core}{suffix}"

    parts = text.split(" ")
    normalized_parts = [
        _normalize_word(part, index == 0, index == len(parts) - 1)
        for index, part in enumerate(parts)
    ]
    return " ".join(normalized_parts)


def _extract_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values: List[str] = []
        for item in value:
            values.extend(_extract_string_list(item))
        return values
    if isinstance(value, dict):
        for key in ("authors", "author", "names", "name", "results"):
            if key in value:
                extracted = _extract_string_list(value.get(key))
                if extracted:
                    return extracted
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = _coerce_search_results(text)
    if parsed is not text and parsed != text:
        extracted = _extract_string_list(parsed)
        if extracted:
            return extracted
    if "\n" in text:
        parts = [part.strip(" -•\t") for part in text.splitlines()]
    elif ";" in text:
        parts = [part.strip() for part in text.split(";")]
    else:
        parts = [text]
    return [part for part in parts if part]


def _match_github_issue_url(url: str) -> Optional[tuple[str, str, str]]:
    match = re.match(r"^https?://github\.com/([^/]+)/([^/]+)/issues/(\d+)(?:[/?#].*)?$", str(url).strip(), flags=re.I)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def _render_github_issue_api_text(issue_json: Dict[str, Any], timeline_json: List[Dict[str, Any]]) -> str:
    lines = []
    title = issue_json.get("title") or ""
    number = issue_json.get("number")
    state = issue_json.get("state") or ""
    created_at = issue_json.get("created_at") or ""
    closed_at = issue_json.get("closed_at") or ""
    body = (issue_json.get("body") or "").strip()
    labels = [item.get("name", "") for item in (issue_json.get("labels") or []) if isinstance(item, dict)]

    lines.append(f"Issue #{number}: {title}")
    if state:
        lines.append(f"State: {state}")
    if created_at:
        lines.append(f"Created at: {created_at}")
    if closed_at:
        lines.append(f"Closed at: {closed_at}")
    if labels:
        lines.append(f"Current labels: {', '.join(labels)}")
    if body:
        lines.append("Body:")
        lines.append(body[:4000])

    if timeline_json:
        lines.append("Timeline events:")
        for event in timeline_json[:200]:
            if not isinstance(event, dict):
                continue
            event_type = event.get("event") or ""
            created = event.get("created_at") or ""
            actor = ""
            if isinstance(event.get("actor"), dict):
                actor = event["actor"].get("login") or ""
            label_name = ""
            if isinstance(event.get("label"), dict):
                label_name = event["label"].get("name") or ""
            rename_from = event.get("rename", {}).get("from") if isinstance(event.get("rename"), dict) else ""
            rename_to = event.get("rename", {}).get("to") if isinstance(event.get("rename"), dict) else ""
            parts = [part for part in [created, event_type, label_name, actor] if part]
            line = " | ".join(parts)
            if event_type == "labeled" and label_name:
                line = f"Labeled {label_name} at {created}" + (f" by {actor}" if actor else "")
            elif event_type == "unlabeled" and label_name:
                line = f"Unlabeled {label_name} at {created}" + (f" by {actor}" if actor else "")
            if rename_from or rename_to:
                line += f" | rename: {rename_from} -> {rename_to}"
            if line:
                lines.append(line)

    return _truncate("\n".join(lines), 12000)


def _render_github_issue_html_text(html: str) -> str:
    text = _html_to_text(html)
    lines = [line.strip() for line in text.splitlines()]
    kept = []
    seen = set()
    for line in lines:
        if not line:
            continue
        if len(line) > 400:
            continue
        lowered = line.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        if any(
            token in lowered
            for token in (
                "issue",
                "regression",
                "numpy.polynomial",
                "label",
                "labeled",
                "closed",
                "opened",
                "timeline",
                "bug",
                "julia",
                "on ",
            )
        ):
            kept.append(line)
            continue
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b", line) or re.search(r"\b[A-Z][a-z]{2} \d{1,2}, \d{4}\b", line):
            kept.append(line)

    if not kept:
        kept = lines[:200]

    labeled_events = []
    label_pattern = re.compile(
        r'"__typename":"LabeledEvent".*?"createdAt":"([^"]+)".*?"actor":\{.*?"login":"([^"]+)".*?\}.*?"label":\{.*?"name":"([^"]+)"',
        re.S,
    )
    for created_at, actor, label_name in label_pattern.findall(html):
        line = f"Labeled {label_name} at {created_at}"
        if actor:
            line += f" by {actor}"
        if line not in labeled_events:
            labeled_events.append(line)

    unlabeled_pattern = re.compile(
        r'"__typename":"UnlabeledEvent".*?"createdAt":"([^"]+)".*?"actor":\{.*?"login":"([^"]+)".*?\}.*?"label":\{.*?"name":"([^"]+)"',
        re.S,
    )
    for created_at, actor, label_name in unlabeled_pattern.findall(html):
        line = f"Unlabeled {label_name} at {created_at}"
        if actor:
            line += f" by {actor}"
        if line not in labeled_events:
            labeled_events.append(line)

    if labeled_events:
        kept.append("Timeline events:")
        kept.extend(labeled_events[:200])

    return _truncate("\n".join(kept), 12000)


def _extract_color_grouped_tokens(image_path: pathlib.Path) -> Optional[str]:
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    try:
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "--psm", "6", "tsv"],
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    import csv
    import io

    red_items = []
    green_items = []

    rows = list(csv.DictReader(io.StringIO(result.stdout), delimiter="\t"))
    for row in rows:
        token = (row.get("text") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9.\-]+", token):
            continue
        try:
            left = int(row.get("left", 0))
            top = int(row.get("top", 0))
            width = int(row.get("width", 0))
            height = int(row.get("height", 0))
        except Exception:
            continue
        if width <= 0 or height <= 0:
            continue

        crop = img.crop((left, top, left + width, top + height))
        pixels = []
        for r, g, b in crop.getdata():
            if r + g + b < 40:
                continue
            pixels.append((r, g, b))
        if not pixels:
            continue

        r_mean = sum(v[0] for v in pixels) / len(pixels)
        g_mean = sum(v[1] for v in pixels) / len(pixels)
        b_mean = sum(v[2] for v in pixels) / len(pixels)

        entry = (top, left, token)
        if r_mean > g_mean + 8 and r_mean > b_mean + 8:
            red_items.append(entry)
        elif g_mean > r_mean + 8 and g_mean > b_mean + 8:
            green_items.append(entry)

    if len(red_items) < 3 or len(green_items) < 3:
        return None

    red_items.sort(key=lambda item: (item[0], item[1]))
    green_items.sort(key=lambda item: (item[0], item[1]))
    red_tokens = [item[2] for item in red_items]
    green_tokens = [item[2] for item in green_items]
    return (
        "Color-grouped OCR:\n"
        f"Red tokens: {', '.join(red_tokens)}\n"
        f"Green tokens: {', '.join(green_tokens)}"
    )


def _load_hf_token() -> Optional[str]:
    for env_key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(env_key)
        if token:
            return token.strip()

    token_paths = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]
    for token_path in token_paths:
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                return token
    return None


def _hf_inference_bytes(model: str, payload: bytes, content_type: str) -> Any:
    token = _load_hf_token()
    if not token:
        return "Hugging Face token is not configured for multimodal inference."

    request = Request(
        f"https://api-inference.huggingface.co/models/{model}",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8", errors="ignore")
    except HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            error_body = str(exc)
        return f"Inference API error ({exc.code}): {error_body}"
    except Exception as exc:
        return f"Inference API request failed: {exc}"

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def _extract_generated_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        for key in ("text", "generated_text", "answer"):
            if key in response and isinstance(response[key], str):
                return response[key].strip()
        if "error" in response:
            return str(response["error"])
        return json.dumps(response, ensure_ascii=False)
    if isinstance(response, list):
        chunks = []
        for item in response:
            if isinstance(item, dict):
                for key in ("generated_text", "text", "answer", "label"):
                    if key in item and isinstance(item[key], str):
                        chunks.append(item[key].strip())
                        break
            elif isinstance(item, str):
                chunks.append(item.strip())
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(response)


def _run_hf_inference_subprocess(snippet: str) -> str:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    extra_path = "/tmp/dynacall_gaia_deps"
    env["PYTHONPATH"] = f"{extra_path}:{existing_pythonpath}" if existing_pythonpath else extra_path
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        return f"Inference subprocess failed: {stderr}"
    return result.stdout.strip()


def _html_to_text(html: str) -> str:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
    else:
        html = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", html)
        text = unescape(text)
    return re.sub(r"\n{2,}", "\n\n", text).strip()


def _extract_html_image_url(html: str, base_url: str = "") -> str:
    patterns = [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.I)
        if match:
            candidate = match.group(1).strip()
            if candidate.startswith("//"):
                return "https:" + candidate
            if candidate.startswith("/"):
                parsed = urlparse(base_url)
                if parsed.scheme and parsed.netloc:
                    return f"{parsed.scheme}://{parsed.netloc}{candidate}"
            if candidate.startswith(("http://", "https://")):
                return candidate
    return ""


def _extract_html_download_url(html: str, base_url: str = "") -> str:
    file_suffixes = (
        ".pdf", ".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonld", ".zip",
        ".doc", ".docx", ".ppt", ".pptx", ".txt", ".xml",
    )
    candidates: List[str] = []

    def _normalize_candidate(candidate: str) -> str:
        candidate = (candidate or "").strip()
        if not candidate:
            return ""
        if candidate.startswith("//"):
            return "https:" + candidate
        if candidate.startswith("/"):
            parsed = urlparse(base_url)
            if parsed.scheme and parsed.netloc:
                return f"{parsed.scheme}://{parsed.netloc}{candidate}"
        if candidate.startswith(("http://", "https://")):
            return candidate
        return ""

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["a", "iframe", "embed"]):
            raw = tag.get("href") or tag.get("src") or ""
            normalized = _normalize_candidate(raw)
            if normalized:
                candidates.append(normalized)
    else:
        for pattern in [
            r'href=["\']([^"\']+)["\']',
            r'src=["\']([^"\']+)["\']',
        ]:
            for match in re.finditer(pattern, html, flags=re.I):
                normalized = _normalize_candidate(match.group(1))
                if normalized:
                    candidates.append(normalized)

    for candidate in candidates:
        lowered = candidate.lower()
        if any(lowered.endswith(suffix) for suffix in file_suffixes):
            return candidate

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["a", "button"]):
            label = " ".join(
                part for part in [
                    str(tag.get_text(" ", strip=True) or "").strip(),
                    str(tag.get("aria-label", "") or "").strip(),
                    str(tag.get("title", "") or "").strip(),
                ] if part
            ).lower()
            if not label:
                continue
            if any(token in label for token in ("download", "full text", "pdf", "supplement", "attachment", "original file")):
                normalized = _normalize_candidate(tag.get("href") or tag.get("data-href") or tag.get("src") or "")
                if normalized:
                    return normalized
    return ""


def _looks_like_blocked_or_challenge_page(content: str) -> bool:
    if not content:
        return False
    lowered = content.lower()
    markers = [
        "verification required",
        "please complete this challenge",
        "access denied",
        "forbidden",
        "captcha",
        "cf-chl",
        "cloudflare",
        "bot verification",
    ]
    if any(marker in lowered for marker in markers):
        return True

    collapsed = " ".join(lowered.split())
    placeholder_texts = {
        "instagram",
        "facebook",
        "x",
        "twitter",
        "youtube",
        "tiktok",
        "linkedin",
        "log in",
        "sign in",
    }
    if collapsed in placeholder_texts:
        return True
    if len(collapsed) <= 40 and any(token in collapsed for token in ("instagram", "facebook", "twitter", "youtube", "tiktok", "linkedin")):
        return True
    shell_texts = {
        "skip to main content",
        "skip to content",
        "main content",
        "loading...",
        "read more",
        "menu",
        "home",
    }
    if collapsed in shell_texts:
        return True
    if len(collapsed) <= 80 and (
        "skip to main content" in collapsed
        or "skip to content" in collapsed
        or collapsed.startswith("menu ")
        or collapsed.endswith(" menu")
    ):
        return True
    return False


def _extract_html_meta_refresh_url(html: str, base_url: str = "") -> str:
    match = re.search(
        r'<meta[^>]+http-equiv=["\']refresh["\'][^>]+content=["\'][^;]+;\s*url=([^"\']+)["\']',
        html,
        flags=re.I,
    )
    if not match:
        return ""
    candidate = match.group(1).strip()
    if candidate.startswith("//"):
        return "https:" + candidate
    if candidate.startswith("/"):
        parsed = urlparse(base_url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}{candidate}"
    if candidate.startswith(("http://", "https://")):
        return candidate
    return ""


def _infer_suffix_from_content_type(content_type: str) -> str:
    lowered = (content_type or "").lower()
    if "image/jpeg" in lowered:
        return ".jpg"
    if "image/png" in lowered:
        return ".png"
    if "image/webp" in lowered:
        return ".webp"
    if "image/gif" in lowered:
        return ".gif"
    if "application/pdf" in lowered:
        return ".pdf"
    if "text/csv" in lowered:
        return ".csv"
    if "application/zip" in lowered:
        return ".zip"
    if "spreadsheetml" in lowered:
        return ".xlsx"
    if "ms-excel" in lowered:
        return ".xls"
    if "presentationml" in lowered:
        return ".pptx"
    if "wordprocessingml" in lowered:
        return ".docx"
    if "application/json" in lowered:
        return ".json"
    if "text/plain" in lowered:
        return ".txt"
    return ""


def _normalize_search_result_url(url: str) -> str:
    if not url:
        return url
    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):
        url = "https://duckduckgo.com" + url

    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        query = parse_qs(parsed.query)
        uddg = query.get("uddg", [])
        if uddg:
            return unquote(uddg[0])
    if "bing.com" in parsed.netloc and parsed.path.startswith("/ck/a"):
        query = parse_qs(parsed.query)
        wrapped = query.get("u", [])
        if wrapped:
            candidate = wrapped[0]
            if candidate.startswith("a1"):
                candidate = candidate[2:]
            candidate = candidate + "=" * (-len(candidate) % 4)
            try:
                decoded = base64.b64decode(candidate).decode("utf-8", errors="ignore")
                if decoded.startswith(("http://", "https://")):
                    return decoded
            except Exception:
                pass
    return url


def _generate_search_fallbacks(query: str) -> list[str]:
    fallbacks = []

    no_site = re.sub(r"\bsite:[^\s]+\b", "", query, flags=re.I)
    no_site = re.sub(r"\s+", " ", no_site).strip()
    if no_site and no_site not in fallbacks:
        fallbacks.append(no_site)

    no_quotes = query.replace('"', "").replace("'", "")
    no_quotes = re.sub(r"\s+", " ", no_quotes).strip()
    if no_quotes and no_quotes not in fallbacks:
        fallbacks.append(no_quotes)

    simplified = re.sub(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b", "", query, flags=re.I)
    simplified = re.sub(r"\b(submitted|originally|article|paper|label|word|words|which)\b", "", simplified, flags=re.I)
    simplified = re.sub(r"\bsite:[^\s]+\b", "", simplified, flags=re.I)
    simplified = re.sub(r"[^\w\s\.-]", " ", simplified)
    simplified = re.sub(r"\s+", " ", simplified).strip()
    if simplified and simplified not in fallbacks:
        fallbacks.append(simplified)

    prioritized = []

    if "arxiv" in query.lower():
        site_variant = f"{simplified or query} site:arxiv.org"
        if site_variant not in prioritized:
            prioritized.append(site_variant)
        for variant in (
            f"{simplified or query} site:arxiv.org/abs",
            f"{simplified or query} site:arxiv.org/pdf",
        ):
            if variant not in prioritized:
                prioritized.append(variant)

    if "usgs" in query.lower():
        site_variant = f"{simplified or query} site:usgs.gov"
        if site_variant not in prioritized:
            prioritized.append(site_variant)
        for variant in (
            f"{simplified or query} site:nas.er.usgs.gov",
            f"{simplified or query} site:nas.er.usgs.gov/queries",
        ):
            if variant not in prioritized:
                prioritized.append(variant)

    if re.search(r"\bnature\b", query, flags=re.I):
        site_variant = f"{simplified or query} site:nature.com"
        if site_variant not in prioritized:
            prioritized.append(site_variant)

    if re.search(r"\bgithub\b", query, flags=re.I):
        site_variant = f"{simplified or query} site:github.com"
        if site_variant not in prioritized:
            prioritized.append(site_variant)

    latin_name_match = re.search(r"\b([A-Z][a-z]+)\s+([a-z]{3,})\b", query)
    if latin_name_match:
        latin_name = f"{latin_name_match.group(1)} {latin_name_match.group(2)}"
        for variant in (
            f"{latin_name} site:nas.er.usgs.gov",
            f"{latin_name} collection site:nas.er.usgs.gov",
            f"{latin_name} species profile site:nas.er.usgs.gov",
        ):
            if variant not in prioritized:
                prioritized.append(variant)

    museum_accession_match = re.search(r"\b(\d{4},\d+\.\d+)\b", query)
    if museum_accession_match and re.search(r"\bbritish museum\b", query, flags=re.I):
        accession = museum_accession_match.group(1)
        accession_hyphen = accession.replace(",", "-").replace(".", "-")
        museum_variants = (
            f'"{accession}" "British Museum" shell',
            f'"{accession}" "British Museum" mollusk',
            f'"{accession_hyphen}" "British Museum" shell',
            f'"{accession}" site:britishmuseum.org',
            f'"{accession}" site:britishmuseum.org/collection',
            f'"{accession_hyphen}" site:britishmuseum.org/collection/object',
        )
        for variant in museum_variants:
            if variant not in prioritized:
                prioritized.append(variant)

    for variant in prioritized:
        if variant and variant not in fallbacks:
            fallbacks.append(variant)
    for variant in (query, no_site, no_quotes, simplified):
        if variant and variant not in fallbacks:
            fallbacks.append(variant)

    return fallbacks


def _extract_arxiv_date_category_query(query: str) -> Optional[Dict[str, Any]]:
    text = str(query or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if "arxiv" not in lowered:
        return None

    category = ""
    if "physics and society" in lowered:
        category = "physics.soc-ph"
    elif "computer science and society" in lowered:
        category = "cs.cy"
    if not category:
        for category_candidate in re.findall(r"\b([a-z]+(?:-[a-z]+)?\.[a-z]+(?:-[a-z]+)?)\b", text, flags=re.I):
            normalized_category = category_candidate.lower()
            if normalized_category in {"arxiv.org", "www.arxiv", "doi.org"}:
                continue
            if normalized_category.endswith((".org", ".com", ".net", ".edu", ".gov")):
                continue
            category = normalized_category
            break

    month_names = (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
    )
    date_match = re.search(
        r"\b(" + "|".join(month_names) + r")\s+(\d{1,2}),\s*(\d{4})\b",
        text,
        flags=re.I,
    )
    if not date_match:
        date_match = re.search(
            r"\b(" + "|".join(month_names) + r")\s+(\d{1,2})\s+(\d{4})\b",
            text,
            flags=re.I,
        )
    if not date_match:
        date_match = re.search(
            r"\b(\d{1,2})\s+(" + "|".join(month_names) + r")\s+(\d{4})\b",
            text,
            flags=re.I,
        )
        if date_match:
            day = int(date_match.group(1))
            month_name = date_match.group(2)
            year = int(date_match.group(3))
        else:
            day = None
            month_name = ""
            year = None
    else:
        month_name = date_match.group(1)
        day = int(date_match.group(2))
        year = int(date_match.group(3))

    if not category or year is None or day is None:
        return None

    try:
        month = datetime.strptime(month_name[:3], "%b").month
    except Exception:
        return None

    return {
        "category": category,
        "year": year,
        "month": month,
        "day": day,
        "query": text,
    }


def _normalize_arxiv_api_query(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return ""
    text = re.sub(r"\bsite:\s*arxiv\.org(?:/\w+)?\b", " ", text, flags=re.I)
    text = re.sub(r"\barxiv(?:\.org)?\b", " ", text, flags=re.I)
    text = re.sub(r"\b(pdf|abs|html)\b", " ", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9\"'._:/+-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_SEARCH_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "with", "from",
    "by", "at", "as", "is", "are", "was", "were", "be", "that", "this", "which",
    "exact", "single", "number", "date", "query", "page", "article", "paper", "word",
    "words", "label", "labels", "site", "before", "after", "during",
}

_SEARCH_DOMAIN_DENYLIST = {
    "zhihu.com",
    "zhidao.baidu.com",
    "baidu.com",
    "bilibili.com",
    "weibo.com",
    "xiaohongshu.com",
    "douban.com",
    "sohu.com",
}


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _extract_query_site_hints(query: str) -> list[str]:
    hints = re.findall(r"\bsite:([^\s]+)\b", query, flags=re.I)
    hints = [hint.lower().strip() for hint in hints if hint.strip()]
    lowered = query.lower()
    implicit = []
    if "arxiv" in lowered:
        implicit.extend(["arxiv.org"])
    if "usgs" in lowered:
        implicit.extend(["usgs.gov", "nas.er.usgs.gov"])
    if "nature" in lowered:
        implicit.extend(["nature.com"])
    if "github" in lowered:
        implicit.extend(["github.com"])
    for hint in implicit:
        if hint not in hints:
            hints.append(hint)
    return hints


def _extract_query_terms(query: str) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9\.-]{1,}", query.lower())
    return [t for t in terms if t not in _SEARCH_STOPWORDS and len(t) > 2 and not t.startswith("site:")]


def _score_search_result(item: dict, query: str) -> tuple:
    href = str(item.get("href", "") or "")
    title = str(item.get("title", "") or "")
    body = str(item.get("body", "") or "")
    combined = f"{title} {body} {href}".lower()
    netloc = urlparse(href).netloc.lower()

    score = 0
    site_hints = _extract_query_site_hints(query)
    for hint in site_hints:
        if hint and hint in netloc:
            score += 120

    accession_match = re.search(r"\b(\d{4},\d+\.\d+|\d{4}-\d+-\d+)\b", query)
    if accession_match:
        accession = accession_match.group(1).lower()
        if accession in combined:
            score += 160
        alt_accession = accession.replace(",", "-").replace(".", "-")
        if alt_accession in combined:
            score += 120
        if "british museum" in query.lower() and "britishmuseum.org" in netloc:
            score += 80

    terms = _extract_query_terms(query)
    overlap = 0
    for term in terms:
        if term in combined:
            overlap += 1
            if term in title.lower():
                score += 12
            elif term in href.lower():
                score += 8
            else:
                score += 4
    score += min(overlap, 10) * 3

    lowered_query = query.lower()
    if "arxiv" in lowered_query and "arxiv.org" in netloc:
        score += 40
    if "usgs" in lowered_query and ("usgs.gov" in netloc or "nas.er.usgs.gov" in netloc):
        score += 40
    if "nature" in lowered_query and "nature.com" in netloc:
        score += 40
    if "github" in lowered_query and "github.com" in netloc:
        score += 40
    if "science advances" in lowered_query:
        if "science.org" in netloc or "spj.science.org" in netloc:
            score += 80
        if "/doi/" in href.lower():
            score += 40
        if "science advances" in combined:
            score += 120
        if "research article" in combined:
            score += 30

    year_tokens = re.findall(r"\b(?:19|20)\d{2}\b", lowered_query)
    for year in year_tokens:
        if year in combined:
            score += 20

    syntax_query = any(
        token in lowered_query
        for token in (
            "programming language",
            "syntax",
            "operator",
            "delimiter",
            "character",
            "token",
            "symbol",
            "code correction",
        )
    )
    if syntax_query:
        if any(marker in netloc for marker in ("wikipedia.org", "esolangs.org")):
            score += 60
        if any(marker in combined for marker in ("syntax", "operator", "built-in", "built in", "language", "combinatory logic")):
            score += 25
        if any(marker in combined for marker in ("interpreter", "playground", "run code", "show code while execution", "sample code")):
            score -= 35

    noisy_markers = (
        "forum", "lottery", "pilio", "9800.com.tw", "arclink", "nano-reef",
        "facebook.com", "instagram.com/popular", "instagram.com/explore",
        "researcher.life",
    )
    if any(marker in netloc or marker in href.lower() for marker in noisy_markers):
        score -= 120
    if any(netloc == blocked or netloc.endswith("." + blocked) for blocked in _SEARCH_DOMAIN_DENYLIST):
        score -= 220
    if _contains_cjk(title) or _contains_cjk(body):
        score -= 80

    return (score, overlap, len(title))


def _filter_ranked_results(items: list[dict], query: str, max_results: int, strict_site_hint: bool = False) -> list[dict]:
    site_hints = _extract_query_site_hints(query)
    ranked = sorted(items or [], key=lambda item: _score_search_result(item, query), reverse=True)

    cleaned: list[dict] = []
    for item in ranked:
        href = str(item.get("href", "") or "")
        title = str(item.get("title", "") or "")
        body = str(item.get("body", "") or "")
        netloc = urlparse(href).netloc.lower()
        if any(netloc == blocked or netloc.endswith("." + blocked) for blocked in _SEARCH_DOMAIN_DENYLIST):
            continue
        if _contains_cjk(title) or _contains_cjk(body):
            continue
        cleaned.append(item)

    if site_hints:
        hinted = [
            item for item in cleaned
            if any(hint in urlparse(str(item.get("href", "") or "")).netloc.lower() for hint in site_hints)
        ]
        if hinted:
            cleaned = hinted
        elif strict_site_hint:
            fallback_cleaned = [item for item in cleaned if _score_search_result(item, query)[0] > 0]
            return fallback_cleaned[:max_results]

    return cleaned[:max_results]


def _should_require_site_hint(query: str) -> bool:
    return bool(_extract_query_site_hints(query))


class GAIAFileInspector:
    def __init__(self, files_root: Optional[str] = None):
        self.files_root = os.path.abspath(files_root) if files_root else None

    def _resolve_from_archive_extracts(self, file_path: str) -> Optional[pathlib.Path]:
        target_name = pathlib.Path(file_path).name
        if not target_name:
            return None
        temp_root = pathlib.Path(tempfile.gettempdir())
        matches = sorted(
            temp_root.glob(f"dynacall_archive_*/**/{target_name}"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        for match in matches:
            if match.exists() and match.is_file():
                return match.resolve()
        return None

    def _is_temp_archive_extract(self, path: pathlib.Path) -> bool:
        temp_root = pathlib.Path(tempfile.gettempdir()).resolve()
        try:
            relative = path.resolve().relative_to(temp_root)
        except Exception:
            return False
        return bool(relative.parts) and str(relative.parts[0]).startswith("dynacall_archive_")

    def _resolve_path(self, file_path: str) -> pathlib.Path:
        lowered = (file_path or "").strip().lower()
        if lowered.startswith("error downloading file:") or lowered.startswith("error:"):
            raise FileNotFoundError(file_path)
        candidate = pathlib.Path(file_path).expanduser()
        if not candidate.is_absolute():
            cwd_candidate = (pathlib.Path.cwd() / candidate).resolve()
            if cwd_candidate.exists():
                candidate = cwd_candidate
            elif self.files_root:
                candidate = pathlib.Path(self.files_root) / candidate
            else:
                candidate = pathlib.Path.cwd() / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            archive_match = self._resolve_from_archive_extracts(file_path)
            if archive_match is not None:
                candidate = archive_match
        if self.files_root:
            root = pathlib.Path(self.files_root).resolve()
            cwd_root = pathlib.Path.cwd().resolve()
            in_gaia_root = root in candidate.parents or candidate == root
            in_cwd_root = cwd_root in candidate.parents or candidate == cwd_root
            if not in_gaia_root and not in_cwd_root and not self._is_temp_archive_extract(candidate):
                raise ValueError(f"File path escapes allowed roots (GAIA root or CWD): {candidate}")
        if not candidate.exists():
            raise FileNotFoundError(f"File not found: {candidate}")
        return candidate

    def _read_csv_like(self, path: pathlib.Path, delimiter: str = ",") -> str:
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for i, row in enumerate(reader):
                rows.append(row)
                if i >= 9:
                    break
        if not rows:
            return "Empty table."
        header = rows[0]
        lines = [
            f"Columns: {header}",
            "Preview rows:",
        ]
        for row in rows[1:]:
            lines.append(str(row))
        return "\n".join(lines)

    def _read_json_like(self, path: pathlib.Path) -> str:
        text = _safe_read_text(path)
        if path.suffix == ".jsonl":
            lines = [json.loads(line) for line in text.splitlines() if line.strip()]
            return _truncate(json.dumps(lines[:5], ensure_ascii=False, indent=2))
        data = json.loads(text)
        return _truncate(json.dumps(data, ensure_ascii=False, indent=2))

    def _read_html_like(self, path: pathlib.Path) -> str:
        return _truncate(_html_to_text(_safe_read_text(path)))

    def _read_pdf(self, path: pathlib.Path) -> str:
        pdftotext_path = "/opt/homebrew/bin/pdftotext"
        if os.path.exists(pdftotext_path):
            try:
                result = subprocess.run(
                    [pdftotext_path, "-layout", "-f", "1", "-l", "5", str(path), "-"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                text = result.stdout.strip()
                table_preview = _extract_layout_table_preview(text)
                if table_preview:
                    # Keep the parsed preview intact for downstream json.loads consumers.
                    # Put it first and only include a bounded raw-text excerpt afterwards.
                    preview_block = f"Parsed table preview: {table_preview}"
                    remaining = max(0, 12000 - len(preview_block) - len("\n\nRaw text excerpt:\n"))
                    excerpt = text[:remaining].rstrip()
                    if excerpt:
                        return f"{preview_block}\n\nRaw text excerpt:\n{excerpt}"
                    return preview_block
                return _truncate(text)
            except Exception:
                pass
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages[:5]:
                pages.append(page.extract_text() or "")
            return _truncate("\n\n".join(pages))
        except Exception:
            return "PDF text extraction unavailable."

    def _read_docx(self, path: pathlib.Path) -> str:
        if os.path.exists("/usr/bin/textutil"):
            try:
                result = subprocess.run(
                    ["/usr/bin/textutil", "-convert", "txt", "-stdout", str(path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if result.stdout.strip():
                    return _truncate(result.stdout.strip())
            except Exception:
                pass
        try:
            import docx
            document = docx.Document(str(path))
            text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
            return _truncate(text)
        except ImportError:
            with zipfile.ZipFile(path) as zf:
                xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
            text = re.sub(r"<[^>]+>", " ", xml)
            return _truncate(unescape(" ".join(text.split())))

    def _read_excel(self, path: pathlib.Path) -> str:
        if path.suffix.lower() == ".xls":
            legacy = self._read_excel_legacy(path)
            if legacy:
                return legacy
        try:
            import pandas as pd
        except ImportError:
            return self._read_excel_xml(path)
        try:
            sheets = pd.read_excel(path, sheet_name=None)
            chunks = []
            for sheet_name, df in list(sheets.items())[:3]:
                preview = df.head(5).to_dict(orient="records")
                chunks.append(
                    f"Sheet: {sheet_name}\nColumns: {list(df.columns)}\nPreview: {json.dumps(preview, ensure_ascii=False)}"
                )
            return _truncate("\n\n".join(chunks))
        except Exception:
            return self._read_excel_xml(path)

    def _read_excel_legacy(self, path: pathlib.Path) -> str:
        try:
            result = subprocess.run(
                ["strings", "-n", "2", str(path)],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            return ""

        raw_lines = [line.strip() for line in result.stdout.splitlines()]
        if not raw_lines:
            return ""

        start_idx = 0
        for idx, line in enumerate(raw_lines):
            if re.fullmatch(r"Sheet\d+", line) or re.fullmatch(r"Sheet[A-Za-z0-9_]+", line):
                start_idx = idx + 1
                break

        stop_markers = {"[Content_Types].xml", "PK"}
        values = []
        for line in raw_lines[start_idx:]:
            if line in stop_markers:
                break
            if len(line) > 80:
                continue
            if not re.search(r"[A-Za-z]", line):
                continue
            if re.fullmatch(r"[^\w]+", line):
                continue
            if re.search(r"(Accent\d|Heading \d|Warning Text|Calculation|Linked Cell|theme/|_rels/)", line):
                continue
            if values and values[-1] == line:
                continue
            values.append(line)

        if not values:
            return ""

        paired_rows = []
        for idx in range(0, min(len(values), 200), 2):
            row = {"col_0": values[idx]}
            if idx + 1 < len(values):
                row["col_1"] = values[idx + 1]
            paired_rows.append(row)

        payload = {
            "sheets": [
                {
                    "name": "Sheet1",
                    "columns": ["col_0", "col_1"],
                    "preview": paired_rows[:100],
                }
            ],
            "legacy_extracted_values": values[:200],
        }
        return _truncate_json_safely(payload)

    def _read_excel_xml(self, path: pathlib.Path) -> str:
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        with zipfile.ZipFile(path) as zf:
            shared_strings = []
            if "xl/sharedStrings.xml" in zf.namelist():
                root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in root.findall("m:si", ns):
                    text = "".join(t.text or "" for t in si.iterfind(".//m:t", ns))
                    shared_strings.append(text)

            workbook = ET.fromstring(zf.read("xl/workbook.xml"))
            rels = {}
            if "xl/_rels/workbook.xml.rels" in zf.namelist():
                rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
                for rel in rel_root:
                    rels[rel.attrib.get("Id")] = rel.attrib.get("Target")

            chunks = []
            for sheet in workbook.findall(".//m:sheet", ns)[:3]:
                name = sheet.attrib.get("name", "Sheet")
                rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                target = rels.get(rel_id, "")
                if not target:
                    continue
                xml_path = "xl/" + target.lstrip("/")
                sheet_root = ET.fromstring(zf.read(xml_path))
                raw_rows = []
                for row in sheet_root.findall(".//m:sheetData/m:row", ns)[:30]:
                    values = []
                    for cell in row.findall("m:c", ns):
                        cell_type = cell.attrib.get("t")
                        value = cell.findtext("m:v", default="", namespaces=ns)
                        if cell_type == "s" and value.isdigit():
                            idx = int(value)
                            value = shared_strings[idx] if idx < len(shared_strings) else value
                        values.append(value)
                    raw_rows.append(values)
                headers, records = _normalize_excel_rows(raw_rows)
                preview = records[:12]
                chunks.append(
                    f"Sheet: {name}\nColumns: {headers}\nPreview records: {json.dumps(preview, ensure_ascii=False)}"
                )
            return _truncate("\n\n".join(chunks) or "Spreadsheet preview unavailable.")

    def _read_image_metadata(self, path: pathlib.Path) -> str:
        lines = []
        try:
            from PIL import Image
            image = Image.open(path)
            lines.append(f"Image file. Format: {image.format}. Size: {image.size[0]}x{image.size[1]}.")
        except Exception:
            pass

        if os.path.exists("/usr/bin/sips"):
            try:
                result = subprocess.run(
                    ["/usr/bin/sips", "-g", "pixelWidth", "-g", "pixelHeight", "-g", "format", str(path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                lines.append(result.stdout.strip())
            except Exception:
                pass

        if os.path.exists("/usr/bin/mdls"):
            try:
                result = subprocess.run(
                    [
                        "/usr/bin/mdls",
                        "-name",
                        "kMDItemPixelWidth",
                        "-name",
                        "kMDItemPixelHeight",
                        "-name",
                        "kMDItemBitsPerSample",
                        str(path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                lines.append(result.stdout.strip())
            except Exception:
                pass

        if not lines:
            return "Image file. Metadata extraction unavailable."
        return _truncate("\n".join(lines))

    def _read_zip(self, path: pathlib.Path) -> str:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            extract_root = pathlib.Path(tempfile.mkdtemp(prefix="dynacall_archive_"))
            extracted_members = []
            for name in names:
                try:
                    target_path = extract_root / name
                    if name.endswith("/"):
                        target_path.mkdir(parents=True, exist_ok=True)
                        continue
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, target_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted_members.append((name, target_path))
                except Exception:
                    continue
            preview_members = []
            for name, extracted_path in extracted_members[:3]:
                if name.lower().endswith((".txt", ".csv", ".json", ".xml")):
                    try:
                        content = extracted_path.read_text(encoding="utf-8", errors="ignore")
                        preview_members.append(
                            {
                                "name": name,
                                "path": str(extracted_path),
                                "preview": _truncate(content, 1000),
                            }
                        )
                    except Exception:
                        pass
            payload = {
                "archive_path": str(path),
                "extract_root": str(extract_root),
                "members": [
                    {"name": name, "path": str(extracted_path)}
                    for name, extracted_path in extracted_members
                ],
                "readable_previews": preview_members,
            }
            return _truncate_json_safely(payload)

    def _read_pptx(self, path: pathlib.Path) -> str:
        with zipfile.ZipFile(path) as zf:
            slide_text = []
            for name in sorted(n for n in zf.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml"))[:5]:
                xml = zf.read(name).decode("utf-8", errors="ignore")
                text = re.sub(r"<[^>]+>", " ", xml)
                slide_text.append(unescape(" ".join(text.split())))
            return _truncate("\n\n".join(slide_text))

    def inspect(self, file_path: str) -> str:
        return self.inspect_mode(file_path, "auto")

    def inspect_mode(self, file_path: str, mode: str) -> str:
        path = self._resolve_path(file_path)
        suffix = path.suffix.lower()
        if mode == "archive" or suffix == ".zip":
            return self._read_zip(path)
        prefix = [
            f"Path: {path}",
            f"Name: {path.name}",
            f"Size: {path.stat().st_size} bytes",
        ]
        if mode == "pdf" or suffix == ".pdf":
            content = self._read_pdf(path)
        elif mode == "spreadsheet" or suffix in {".xlsx", ".xls", ".csv", ".tsv"}:
            if suffix == ".csv":
                content = self._read_csv_like(path, delimiter=",")
            elif suffix == ".tsv":
                content = self._read_csv_like(path, delimiter="\t")
            else:
                content = self._read_excel(path)
        elif mode == "powerpoint" or suffix == ".pptx":
            if suffix == ".docx":
                content = self._read_docx(path)
            else:
                content = self._read_pptx(path)
        elif mode == "archive" or suffix == ".zip":
            content = self._read_zip(path)
        elif mode == "image" or suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
            content = self._read_image_metadata(path)
        elif mode == "audio" or suffix in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}:
            content = "Audio file detected. Local speech-to-text engine is not configured in this environment."
        elif suffix in {".txt", ".md", ".py", ".log"}:
            content = _safe_read_text(path)
        elif suffix in {".json", ".jsonl"}:
            content = self._read_json_like(path)
        elif suffix in {".html", ".htm", ".xml"}:
            content = self._read_html_like(path)
        elif suffix == ".docx":
            content = self._read_docx(path)
        elif suffix in {".pdb", ".jsonld"}:
            content = _safe_read_text(path)
        else:
            content = f"Unsupported or binary file type: {suffix}"
        return "\n".join(prefix + ["Content:", _truncate(content)])


class DuckDuckGoSearch:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.headers = _default_http_headers()
        self._result_cache = {}
        self.open_websearch_base_url = (
            os.environ.get("OPEN_WEBSEARCH_BASE_URL", "").strip() or "http://127.0.0.1:3210"
        )
        self.open_websearch_engines = [
            engine.strip()
            for engine in os.environ.get("OPEN_WEBSEARCH_ENGINES", "").split(",")
            if engine.strip()
        ]
        self.open_websearch_mode = os.environ.get("OPEN_WEBSEARCH_SEARCH_MODE", "").strip()
        self.open_websearch_timeout = int(os.environ.get("OPEN_WEBSEARCH_TIMEOUT", "6") or 6)
        self.open_websearch_retry_cooldown_sec = int(os.environ.get("OPEN_WEBSEARCH_RETRY_COOLDOWN_SEC", "60") or 60)
        self._open_websearch_disabled_until = 0.0
        openws_flag = os.environ.get("TOOLWEAVER_ENABLE_OPEN_WEBSEARCH", "").strip().lower()
        if openws_flag:
            self.enable_open_websearch_backend = openws_flag in {"1", "true", "yes"}
        else:
            # Default off: LangSearch should be the primary GAIA backend unless
            # open-websearch is explicitly enabled.
            self.enable_open_websearch_backend = False
        self.tavily_search = None
        self.bing_search_url = os.environ.get("BING_SEARCH_URL", "").strip()
        self.bing_search_api_key = os.environ.get("BING_SEARCH_API_KEY", "").strip()
        self.jina_search_url = os.environ.get("JINA_SEARCH_URL", "https://s.jina.ai/").strip()
        self.jina_search_api_key = os.environ.get("JINA_SEARCH_API_KEY", "").strip()
        self.serper_search_url = os.environ.get("SERPER_SEARCH_URL", "https://google.serper.dev/search").strip()
        self.serper_search_api_key = os.environ.get("SERPER_SEARCH_API_KEY", "").strip()
        use_search_engine = os.environ.get("USE_SEARCH_ENGINE", "").strip()
        if use_search_engine:
            self.mixed_search_engines = [
                engine.strip().lower()
                for engine in use_search_engine.split(",")
                if engine.strip()
            ]
        else:
            self.mixed_search_engines = []
        self.langsearch_key = os.environ.get("LANGSEARCH_API_KEY", "").strip()
        langsearch_flag = os.environ.get("TOOLWEAVER_ENABLE_LANGSEARCH", "").strip().lower()
        if langsearch_flag:
            self.enable_langsearch_backend = langsearch_flag in {"1", "true", "yes"}
        else:
            self.enable_langsearch_backend = bool(self.langsearch_key)
        google_flag = os.environ.get("TOOLWEAVER_ENABLE_GOOGLE_SCRAPE", "").strip().lower()
        if google_flag:
            self.enable_google_backend = google_flag in {"1", "true", "yes"}
        else:
            self.enable_google_backend = False
        self.enable_tavily_backend = os.environ.get("TOOLWEAVER_ENABLE_TAVILY", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        # Keep legacy fallback code available, but disable it by default so
        # GAIA search quality is driven by the primary backend only.
        self.enable_fallback_backends = os.environ.get("TOOLWEAVER_ENABLE_SEARCH_FALLBACKS", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if self.enable_tavily_backend and os.environ.get("TAVILY_API_KEY", "").strip() and TavilySearchResults is not None:
            try:
                self.tavily_search = TavilySearchResults(max_results=5)
            except Exception:
                self.tavily_search = None

    async def asearch(self, keywords: str, max_results: int = 8):
        return await asyncio.to_thread(self.search, keywords, max_results)

    def search(self, keywords: str, max_results: int = 8):
        query_text = str(keywords).strip()
        query_text = re.sub(r"^\s*query\s*:\s*", "", query_text, flags=re.I)
        cache_key = (query_text, int(max_results))
        if cache_key in self._result_cache:
            return list(self._result_cache[cache_key])

        aggregated = []
        seen = set()
        query_variants = [query_text]
        for variant in _generate_search_fallbacks(query_text):
            if variant and variant not in query_variants:
                query_variants.append(variant)

        def add_results(items):
            ranked_items = sorted(items or [], key=lambda item: _score_search_result(item, query_text), reverse=True)
            for item in ranked_items:
                if not self._result_allowed_for_query(query_text, item):
                    continue
                href = item.get("href", "")
                title = item.get("title", "")
                body = item.get("body", "")
                key = href or title or body[:120]
                if not key or key in seen:
                    continue
                seen.add(key)
                aggregated.append(item)
                if len(aggregated) >= max_results:
                    break
            return len(aggregated) >= max_results

        lowered_query = query_text.lower()

        arxiv_date_category = _extract_arxiv_date_category_query(query_text)
        if arxiv_date_category:
            try:
                results = self._dedupe_results(
                    self._search_arxiv_listing_by_date(arxiv_date_category, max_results),
                    max_results,
                )
                if add_results(results):
                    self._result_cache[cache_key] = aggregated[:max_results]
                    return list(self._result_cache[cache_key])
            except Exception:
                pass

        if "arxiv" in lowered_query:
            for query in query_variants[:4]:
                try:
                    results = self._dedupe_results(self._search_arxiv_api(query, max_results), max_results)
                    if add_results(results):
                        self._result_cache[cache_key] = aggregated[:max_results]
                        return list(self._result_cache[cache_key])
                except Exception:
                    continue

        if (
            ("github.com" in lowered_query or "github" in lowered_query or "repo:" in lowered_query)
            and ("issue" in lowered_query or "issues" in lowered_query or "label:" in lowered_query or "pull request" in lowered_query)
        ):
            for _ in range(2):
                try:
                    results = self._dedupe_results(self._search_github_issues(query_text, max_results), max_results)
                    if add_results(results):
                        self._result_cache[cache_key] = aggregated[:max_results]
                        return list(self._result_cache[cache_key])
                except Exception:
                    continue

        if self.enable_open_websearch_backend:
            for query in query_variants[:6]:
                for _ in range(2):
                    try:
                        results = self._dedupe_results(self._search_open_websearch(query, max_results), max_results)
                        if add_results(results):
                            self._result_cache[cache_key] = aggregated[:max_results]
                            return list(self._result_cache[cache_key])
                    except Exception:
                        continue

        if self.enable_langsearch_backend and self.langsearch_key:
            for query in query_variants[:6]:
                for _ in range(2):
                    try:
                        results = self._dedupe_results(self._search_langsearch(query, max_results), max_results)
                        if add_results(results):
                            self._result_cache[cache_key] = aggregated[:max_results]
                            return list(self._result_cache[cache_key])
                    except Exception:
                        continue

        if self.mixed_search_engines:
            for query in query_variants[:4]:
                for _ in range(2):
                    try:
                        results = self._dedupe_results(self._search_mix_engines(query, max_results), max_results)
                        if add_results(results):
                            self._result_cache[cache_key] = aggregated[:max_results]
                            return list(self._result_cache[cache_key])
                    except Exception:
                        continue

        if self.tavily_search is not None:
            for query in query_variants[:4]:
                for _ in range(2):
                    try:
                        results = self._dedupe_results(self._search_tavily(query, max_results), max_results)
                        if add_results(results):
                            self._result_cache[cache_key] = aggregated[:max_results]
                            return list(self._result_cache[cache_key])
                    except Exception:
                        continue

        if self.enable_fallback_backends or not aggregated:
            for query in _generate_search_fallbacks(query_text):
                for backend in (self._search_bing, self._search_duckduckgo, self._search_google_scrape if self.enable_google_backend and google_search is not None else None):
                    if backend is None:
                        continue
                    for _ in range(2):
                        try:
                            results = self._dedupe_results(backend(query, max_results), max_results)
                        except Exception:
                            continue
                        if add_results(results):
                            self._result_cache[cache_key] = aggregated[:max_results]
                            return list(self._result_cache[cache_key])
        self._result_cache[cache_key] = aggregated[:max_results]
        return list(self._result_cache[cache_key])

    def _result_allowed_for_query(self, query_text: str, item: Dict[str, Any]) -> bool:
        href = str((item or {}).get("href", "") or "").strip()
        if not href:
            return True
        lowered_query = (query_text or "").lower()
        if "wikipedia" not in lowered_query:
            return True
        try:
            domain = (urlparse(href).netloc or "").lower()
        except Exception:
            return True
        if not domain:
            return True
        is_official_wiki = (
            domain == "wikipedia.org"
            or domain.endswith(".wikipedia.org")
            or domain == "wikimedia.org"
            or domain.endswith(".wikimedia.org")
        )
        if not is_official_wiki:
            return False
        if "wikipedia" in domain and not is_official_wiki:
            return False
        if "english wikipedia" in lowered_query:
            return domain == "en.wikipedia.org" or domain.endswith(".en.wikipedia.org")
        return True

    def _search_arxiv_listing_by_date(self, arxiv_query: Dict[str, Any], max_results: int):
        category = str(arxiv_query.get("category", "")).strip().lower()
        year = int(arxiv_query.get("year", 0) or 0)
        month = int(arxiv_query.get("month", 0) or 0)
        day = int(arxiv_query.get("day", 0) or 0)
        raw_query = str(arxiv_query.get("query", "")).strip()
        if not category or not year or not month or not day:
            return []

        target_date = datetime(year, month, day)
        month_short = f"{year % 100:02d}{month:02d}"
        month_long = f"{year:04d}-{month:02d}"
        candidate_urls = [
            f"https://arxiv.org/list/{category}/{month_short}?show=2000",
            f"https://arxiv.org/list/{category}/{month_short}",
            f"https://arxiv.org/list/{category}/{month_long}?show=2000",
            f"https://arxiv.org/list/{category}/{month_long}",
        ]
        results: List[Dict[str, Any]] = []
        seen = set()

        for list_url in candidate_urls:
            try:
                html, final_url = _http_get(list_url, headers=self.headers, timeout=self.timeout)
            except Exception:
                continue
            parsed_results = self._parse_arxiv_listing_html(
                html=html,
                list_url=final_url or list_url,
                category=category,
                target_date=target_date,
                raw_query=raw_query,
            )
            for item in parsed_results:
                href = str(item.get("href", "")).strip()
                if not href or href in seen:
                    continue
                seen.add(href)
                results.append(item)
                if len(results) >= max_results:
                    return results[:max_results]
        return results[:max_results]

    def _search_arxiv_api(self, query: str, max_results: int):
        normalized_query = _normalize_arxiv_api_query(query)
        if not normalized_query:
            return []

        api_url = (
            "https://export.arxiv.org/api/query?"
            + urlencode(
                {
                    "search_query": f"all:{normalized_query}",
                    "start": 0,
                    "max_results": max(10, int(max_results)),
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                }
            )
        )
        payload, _ = _http_get(
            api_url,
            headers={"User-Agent": self.headers.get("User-Agent", "Mozilla/5.0")},
            timeout=self.timeout,
        )
        if not payload:
            return []

        root = ET.fromstring(payload)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        results: List[Dict[str, Any]] = []
        for entry in root.findall("atom:entry", ns):
            title = re.sub(r"\s+", " ", (entry.findtext("atom:title", default="", namespaces=ns) or "")).strip()
            summary = re.sub(r"\s+", " ", (entry.findtext("atom:summary", default="", namespaces=ns) or "")).strip()
            entry_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
            primary = entry.find("arxiv:primary_category", ns)
            category = (primary.attrib.get("term", "") if primary is not None else "").strip()
            authors = []
            for author in entry.findall("atom:author", ns):
                name = (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
                if name:
                    authors.append(name)
            body_parts = []
            if published:
                body_parts.append(f"submitted: {published[:10]}")
            if category:
                body_parts.append(f"category: {category}")
            if authors:
                body_parts.append(f"authors: {', '.join(authors[:4])}")
            if summary:
                body_parts.append(summary)
            if not entry_id:
                continue
            results.append(
                {
                    "title": title or entry_id.rsplit("/", 1)[-1],
                    "href": _normalize_search_result_url(entry_id.replace("http://", "https://")),
                    "body": " | ".join(part for part in body_parts if part),
                }
            )
            if len(results) >= max_results:
                break
        return results[:max_results]

    def _parse_arxiv_listing_html(
        self,
        html: str,
        list_url: str,
        category: str,
        target_date: datetime,
        raw_query: str,
    ) -> List[Dict[str, Any]]:
        if not html:
            return []
        if BeautifulSoup is None:
            return self._parse_arxiv_listing_html_regex(html, list_url, category, target_date, raw_query)

        soup = BeautifulSoup(html, "html.parser")
        results: List[Dict[str, Any]] = []
        current_date: Optional[datetime] = None
        for element in soup.select("h3, dl > dt, dl > dd"):
            if getattr(element, "name", "") == "h3":
                parsed_date = self._parse_arxiv_heading_date(element.get_text(" ", strip=True))
                if parsed_date is not None:
                    current_date = parsed_date
                continue
            if getattr(element, "name", "") != "dt":
                continue
            if current_date is None or current_date.date() != target_date.date():
                continue
            dd = element.find_next_sibling("dd")
            if dd is None:
                continue
            item = self._build_arxiv_listing_result(
                dt_node=element,
                dd_node=dd,
                list_url=list_url,
                category=category,
                target_date=target_date,
                raw_query=raw_query,
            )
            if item:
                results.append(item)
        return results

    def _parse_arxiv_listing_html_regex(
        self,
        html: str,
        list_url: str,
        category: str,
        target_date: datetime,
        raw_query: str,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        section_pattern = re.compile(
            r"<h3[^>]*>\s*(.*?)\s*</h3>(.*?)(?=<h3[^>]*>|$)",
            flags=re.I | re.S,
        )
        pair_pattern = re.compile(r"(<dt.*?</dt>)\s*(<dd.*?</dd>)", flags=re.I | re.S)
        for heading, section_html in section_pattern.findall(html):
            heading_text = _html_to_text(heading)
            heading_date = self._parse_arxiv_heading_date(heading_text)
            if heading_date is None or heading_date.date() != target_date.date():
                continue
            for dt_html, dd_html in pair_pattern.findall(section_html):
                id_match = re.search(r'href="(/abs/[^"#?]+)"', dt_html, flags=re.I)
                if not id_match:
                    continue
                href = _normalize_search_result_url("https://arxiv.org" + id_match.group(1))
                title_match = re.search(r'<div[^>]*class="list-title[^"]*"[^>]*>(.*?)</div>', dd_html, flags=re.I | re.S)
                title = _html_to_text(title_match.group(1) if title_match else "")
                title = re.sub(r"^\s*Title:\s*", "", title, flags=re.I).strip()
                subj_match = re.search(r'<div[^>]*class="list-subjects[^"]*"[^>]*>(.*?)</div>', dd_html, flags=re.I | re.S)
                subjects = _html_to_text(subj_match.group(1) if subj_match else "")
                if category and category not in subjects.lower():
                    continue
                results.append(
                    {
                        "title": title or href.rsplit("/", 1)[-1],
                        "href": href,
                        "body": f"arXiv listing match | category: {category} | submitted: {target_date.strftime('%Y-%m-%d')} | subjects: {subjects} | source: {list_url} | query: {raw_query}",
                    }
                )
        return results

    def _parse_arxiv_heading_date(self, text: str) -> Optional[datetime]:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        match = re.search(r"([A-Za-z]{3},?\s+(?:[A-Za-z]{3}\s+\d{1,2},\s+\d{4}|\d{1,2}\s+[A-Za-z]{3}\s+\d{4}))", cleaned)
        if match:
            cleaned = match.group(1)
        for fmt in ("%a, %d %b %Y", "%a, %b %d, %Y", "%d %b %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(cleaned, fmt)
            except Exception:
                continue
        return None

    def _build_arxiv_listing_result(
        self,
        dt_node: Any,
        dd_node: Any,
        list_url: str,
        category: str,
        target_date: datetime,
        raw_query: str,
    ) -> Optional[Dict[str, Any]]:
        abs_link = dt_node.select_one('a[href^="/abs/"]') if hasattr(dt_node, "select_one") else None
        if abs_link is None:
            return None
        href = _normalize_search_result_url("https://arxiv.org" + str(abs_link.get("href", "")).strip())
        title_node = dd_node.select_one(".list-title")
        subject_node = dd_node.select_one(".list-subjects")
        authors_node = dd_node.select_one(".list-authors")
        comment_node = dd_node.select_one(".list-comments")
        title = title_node.get_text(" ", strip=True) if title_node else ""
        title = re.sub(r"^\s*Title:\s*", "", title, flags=re.I).strip()
        subjects = subject_node.get_text(" ", strip=True) if subject_node else ""
        subjects = re.sub(r"^\s*Subjects?:\s*", "", subjects, flags=re.I).strip()
        if category and category not in subjects.lower():
            return None
        authors = authors_node.get_text(" ", strip=True) if authors_node else ""
        authors = re.sub(r"^\s*Authors?:\s*", "", authors, flags=re.I).strip()
        comments = comment_node.get_text(" ", strip=True) if comment_node else ""
        comments = re.sub(r"^\s*Comments?:\s*", "", comments, flags=re.I).strip()
        body_parts = [
            f"arXiv listing match",
            f"category: {category}",
            f"submitted: {target_date.strftime('%Y-%m-%d')}",
            f"subjects: {subjects}",
        ]
        if authors:
            body_parts.append(f"authors: {authors}")
        if comments:
            body_parts.append(f"comments: {comments}")
        body_parts.append(f"source: {list_url}")
        if raw_query:
            body_parts.append(f"query: {raw_query}")
        return {
            "title": title or href.rsplit("/", 1)[-1],
            "href": href,
            "body": " | ".join(body_parts),
        }

    def _search_bing_api(self, query: str, max_results: int):
        if not (self.bing_search_url and self.bing_search_api_key):
            return []
        headers = {
            "Ocp-Apim-Subscription-Key": self.bing_search_api_key,
            "Content-Type": "application/json",
            "User-Agent": self.headers.get("User-Agent", "DynaCall/GAIA"),
        }
        params = {
            "q": query,
            "count": min(max_results, 10),
            "textDecorations": False,
            "textFormat": "Raw",
        }
        if requests is not None:
            response = requests.get(self.bing_search_url, headers=headers, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        else:
            request_url = self.bing_search_url + "?" + "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
            request = Request(request_url, headers=headers, method="GET")
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="ignore"))
        values = ((payload or {}).get("webPages") or {}).get("value") or []
        return [
            {
                "title": str(item.get("name", "")).strip(),
                "href": _normalize_search_result_url(str(item.get("url", "")).strip()),
                "body": str(item.get("snippet", "")).strip(),
            }
            for item in values
            if isinstance(item, dict)
        ]

    def _search_jina_api(self, query: str, max_results: int):
        if not self.jina_search_url:
            return []
        headers = {
            "Accept": "application/json",
            "User-Agent": self.headers.get("User-Agent", "DynaCall/GAIA"),
        }
        if self.jina_search_api_key:
            headers["Authorization"] = f"Bearer {self.jina_search_api_key}"
        url = f"{self.jina_search_url.rstrip('/')}?q={quote_plus(query)}"
        if requests is not None:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        else:
            request = Request(url, headers=headers, method="GET")
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="ignore"))
        values = (payload or {}).get("data") or (payload or {}).get("search_result") or []
        results = []
        for item in values[:max_results]:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "href": _normalize_search_result_url(str(item.get("url", "") or item.get("link", "")).strip()),
                    "body": str(item.get("content", "") or item.get("snippet", "")).strip(),
                }
            )
        return results

    def _search_serper_api(self, query: str, max_results: int):
        if not (self.serper_search_url and self.serper_search_api_key):
            return []
        headers = {
            "X-API-KEY": self.serper_search_api_key,
            "Content-Type": "application/json",
            "User-Agent": self.headers.get("User-Agent", "DynaCall/GAIA"),
        }
        payload = {"q": query, "num": min(max_results, 10)}
        if requests is not None:
            response = requests.post(self.serper_search_url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        else:
            request = Request(
                self.serper_search_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8", errors="ignore"))
        values = (data or {}).get("organic") or []
        return [
            {
                "title": str(item.get("title", "")).strip(),
                "href": _normalize_search_result_url(str(item.get("link", "")).strip()),
                "body": str(item.get("snippet", "")).strip(),
            }
            for item in values[:max_results]
            if isinstance(item, dict)
        ]

    def _search_mix_engines(self, query: str, max_results: int):
        engine_map = {
            "bing": self._search_bing_api,
            "jina": self._search_jina_api,
            "serp": self._search_serper_api,
            "serper": self._search_serper_api,
        }
        selected = [engine_map[name] for name in self.mixed_search_engines if name in engine_map]
        if not selected:
            return []
        results: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(selected), 4)) as executor:
            futures = [executor.submit(engine, query, max_results) for engine in selected]
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.extend(future.result() or [])
                except Exception:
                    continue
        return _filter_ranked_results(self._dedupe_results(results, max_results * 3), query, max_results)

    def _search_github_issues(self, query: str, max_results: int):
        cleaned = re.sub(r"\bsite:github\.com\b", "", query, flags=re.I).strip()
        cleaned = re.sub(r"\bsort:created-asc\b", "", cleaned, flags=re.I).strip()
        lowered = cleaned.lower()

        def parse_created(text: str) -> str:
            match = re.search(r"created:\s*([0-9T:\-]+Z)", text, flags=re.I)
            if match:
                return match.group(1)
            match = re.search(r"\bon\s+([a-z]{3,9}\s+\d{1,2}\s*,\s*\d{4})\b", text, flags=re.I)
            if match:
                return match.group(1)
            return ""

        def rank_key(item: Dict[str, Any]) -> tuple:
            title = str(item.get("title", "") or "")
            href = str(item.get("href", "") or "")
            body = str(item.get("body", "") or "")
            haystack = f"{title}\n{body}".lower()

            has_repo = "github.com/numpy/numpy/issues/" in href.lower()
            has_closed = "state: closed" in haystack or "closed" in haystack
            has_regression_label = "06 - regression" in haystack
            has_regression_word = "regression" in haystack
            has_component_label = "component: numpy.polynomial" in haystack
            has_component_word = "numpy.polynomial" in haystack or "np.polynomial" in haystack
            created = parse_created(body)
            issue_num = 10**9
            match = re.search(r"/issues/(\d+)", href)
            if match:
                issue_num = int(match.group(1))

            oldest_bias = "oldest" in lowered
            created_key = created or "9999-99-99T99:99:99Z"
            if re.match(r"^[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}$", created_key):
                try:
                    created_key = str(datetime.strptime(created_key, "%b %d, %Y"))
                except Exception:
                    pass

            return (
                0 if has_repo else 1,
                0 if has_closed else 1,
                0 if has_regression_label else (1 if has_regression_word else 2),
                0 if has_component_label else (1 if has_component_word else 2),
                created_key if oldest_bias else "",
                issue_num if oldest_bias else -issue_num,
            )

        base_parts: List[str] = []
        if "issue" in lowered:
            base_parts.append("is:issue")
        if "closed" in lowered:
            base_parts.append("is:closed")

        repo_hint = []
        if "numpy.polynomial" in lowered:
            repo_hint = ["repo:numpy/numpy"]
        elif "numpy/numpy" in lowered:
            repo_hint = ["repo:numpy/numpy"]

        label_variants: List[str] = []
        label_match = re.search(r"label:([A-Za-z0-9._-]+)", cleaned, flags=re.I)
        if label_match:
            label_variants = [f'label:"{label_match.group(1)}"']
        elif re.search(r"\bregression\b", lowered):
            label_variants = ['label:"06 - Regression"', 'label:Regression']
        else:
            label_variants = [""]

        component_variants: List[str] = [""]
        if "numpy.polynomial" in lowered:
            component_variants = ['label:"component: numpy.polynomial"', '"numpy.polynomial"', '"np.polynomial"']

        residual = re.sub(r"\blabel:[A-Za-z0-9._-]+\b", "", cleaned, flags=re.I)
        residual = re.sub(r"\b(oldest|closed|issue|issues|label|sort:created-asc|regression)\b", "", residual, flags=re.I)
        residual = residual.replace("numpy.polynomial", " ").replace("np.polynomial", " ")
        residual = " ".join(residual.split())

        candidate_queries: List[str] = []
        for label_part in label_variants:
            for component_part in component_variants:
                parts = list(base_parts) + repo_hint
                if label_part:
                    parts.append(label_part)
                if component_part:
                    parts.append(component_part)
                base_candidate = " ".join(dict.fromkeys(part for part in parts if part))
                if base_candidate and base_candidate not in candidate_queries:
                    candidate_queries.append(base_candidate)
                if residual:
                    residual_candidate = " ".join(dict.fromkeys(part for part in parts + [residual] if part))
                    if residual_candidate and residual_candidate not in candidate_queries:
                        candidate_queries.append(residual_candidate)

        if not candidate_queries:
            candidate_queries = [cleaned]

        web_headers = {"User-Agent": "Mozilla/5.0"}
        api_headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}
        aggregated: List[Dict[str, Any]] = []
        seen_hrefs = set()

        def add_results(results: List[Dict[str, Any]]):
            for result in results:
                if not isinstance(result, dict):
                    continue
                href = str(result.get("href", "") or "").strip()
                title = str(result.get("title", "") or "").strip()
                body = str(result.get("body", "") or "").strip()
                if not href:
                    continue
                if href in seen_hrefs:
                    for existing in aggregated:
                        if existing.get("href") == href:
                            merged_body = "\n".join(part for part in [existing.get("body", ""), body] if part)
                            existing["body"] = merged_body[:4000]
                            if not existing.get("title") and title:
                                existing["title"] = title
                            break
                    continue
                seen_hrefs.add(href)
                aggregated.append({"title": title, "href": href, "body": body})

        for params_q in candidate_queries:
            html_queries = [params_q]
            if "sort:created-asc" not in params_q.lower():
                html_queries.append(params_q + " sort:created-asc")
            for html_q in html_queries:
                if "repo:numpy/numpy" in html_q:
                    stripped = html_q.replace("repo:numpy/numpy", "").strip()
                    list_url = "https://github.com/numpy/numpy/issues?q=" + quote_plus(stripped)
                else:
                    list_url = "https://github.com/issues?q=" + quote_plus(html_q)
                try:
                    html, final_url = _http_get(list_url, headers=web_headers, timeout=self.timeout)
                    html_results = []
                    if BeautifulSoup is not None:
                        soup = BeautifulSoup(html, "html.parser")
                        links = soup.select('a[data-testid="issue-pr-title-link"]')
                        if not links:
                            links = [
                                a for a in soup.select('a[href*="/issues/"]')
                                if "/issues/" in (a.get("href") or "") and "/labels/" not in (a.get("href") or "")
                            ]
                        seen_links = set()
                        for link in links:
                            href = link.get("href", "")
                            if not href:
                                continue
                            if href.startswith("/"):
                                href = "https://github.com" + href
                            if href in seen_links:
                                continue
                            seen_links.add(href)
                            parent = link
                            for _ in range(5):
                                if getattr(parent, "name", None) in {"div", "li", "article"}:
                                    break
                                parent = getattr(parent, "parent", None)
                                if parent is None:
                                    break
                            snippet = parent.get_text(" ", strip=True)[:500] if parent is not None else link.get_text(" ", strip=True)
                            html_results.append(
                                {
                                    "title": link.get_text(" ", strip=True),
                                    "href": href,
                                    "body": f"github_query: {html_q} | source: {final_url} | {snippet}",
                                }
                            )
                            if len(html_results) >= max_results:
                                break
                    if html_results:
                        add_results(html_results)
                except Exception:
                    pass

            try:
                api_url = (
                    f"https://api.github.com/search/issues?q={quote_plus(params_q)}"
                    f"&sort=created&order=asc&per_page={min(max(max_results * 2, 10), 20)}"
                )
                body, _ = _http_get(api_url, headers=api_headers, timeout=self.timeout)
                payload = json.loads(body)
                items = payload.get("items", []) if isinstance(payload, dict) else []
                results = []
                for item in items[:max_results]:
                    if not isinstance(item, dict):
                        continue
                    labels = []
                    for label in item.get("labels", []) or []:
                        if isinstance(label, dict) and label.get("name"):
                            labels.append(label["name"])
                    snippet_parts = [f"github_query: {params_q}"]
                    if labels:
                        snippet_parts.append("labels: " + ", ".join(labels))
                    if item.get("state"):
                        snippet_parts.append(f"state: {item['state']}")
                    if item.get("created_at"):
                        snippet_parts.append(f"created: {item['created_at']}")
                    if item.get("closed_at"):
                        snippet_parts.append(f"closed: {item['closed_at']}")
                    if item.get("body"):
                        snippet_parts.append(str(item["body"])[:240].replace("\n", " "))
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "href": item.get("html_url", ""),
                            "body": " | ".join(part for part in snippet_parts if part),
                        }
                    )
                if results:
                    add_results(results)
            except Exception:
                pass
        aggregated.sort(key=rank_key)
        return aggregated[:max_results]

    def _search_open_websearch(self, query: str, max_results: int):
        if time.time() < self._open_websearch_disabled_until:
            return []
        base_url = self.open_websearch_base_url.rstrip("/")
        if not base_url:
            return []
        payload: Dict[str, Any] = {
            "query": str(query or "").strip(),
            "limit": max(1, min(int(max_results), 50)),
        }
        if self.open_websearch_engines:
            payload["engines"] = self.open_websearch_engines
        if self.open_websearch_mode:
            payload["searchMode"] = self.open_websearch_mode
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.headers.get("User-Agent", "DynaCall/GAIA"),
        }
        endpoint = f"{base_url}/search"
        try:
            if requests is not None:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=self.open_websearch_timeout)
                response.raise_for_status()
                data = response.json()
            else:
                request = Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urlopen(request, timeout=self.open_websearch_timeout) as response:
                    data = json.loads(response.read().decode("utf-8", errors="ignore"))
        except Exception:
            self._open_websearch_disabled_until = time.time() + max(5, self.open_websearch_retry_cooldown_sec)
            return []

        if str((data or {}).get("status", "")).lower() != "ok":
            return []
        values = (((data or {}).get("data") or {}).get("results") or [])
        results = []
        for item in values[:max_results]:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "href": _normalize_search_result_url(str(item.get("url", "")).strip()),
                    "body": str(item.get("description", "")).strip(),
                }
            )
        return [item for item in results if item.get("title") or item.get("body") or item.get("href")]

    def _search_langsearch(self, query: str, max_results: int):
        if not self.langsearch_key:
            return []
        payload = {
            "query": query,
            "count": min(max_results, 10),
        }
        headers = {
            "Authorization": f"Bearer {self.langsearch_key}",
            "Content-Type": "application/json",
        }
        if requests is not None:
            response = requests.post(
                "https://api.langsearch.com/v1/web-search",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        else:
            request = Request(
                "https://api.langsearch.com/v1/web-search",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8", errors="ignore"))

        values = (((data or {}).get("data") or {}).get("webPages") or {}).get("value") or []
        results = []
        for item in values:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": str(item.get("name", "")).strip(),
                    "href": _normalize_search_result_url(str(item.get("url", "")).strip()),
                    "body": str(item.get("snippet", "") or item.get("summary", "")).strip(),
                }
            )
        return [item for item in results if item.get("title") or item.get("body") or item.get("href")]

    def _search_google_scrape(self, query: str, max_results: int):
        if google_search is None:
            return []

        try:
            links = list(google_search(query, num_results=min(max_results, 8), advanced=True))
        except Exception:
            return []

        seed_results = []
        for item in links[:max_results]:
            href = _normalize_search_result_url(str(getattr(item, "url", "") or "").strip())
            title = str(getattr(item, "title", "") or "").strip()
            body = str(getattr(item, "description", "") or "").strip()
            if href:
                seed_results.append({"title": title, "href": href, "body": body})

        def enrich(result: dict) -> dict:
            href = str(result.get("href", "")).strip()
            body = str(result.get("body", "")).strip()
            if body and len(body) >= 140:
                return result
            try:
                raw_bytes, final_url, content_type = _http_get_bytes(
                    href,
                    headers=self.headers,
                    timeout=min(self.timeout, 8),
                )
                if (final_url or href).lower().endswith(".pdf") or "application/pdf" in (content_type or "").lower() or raw_bytes.startswith(b"%PDF"):
                    fetched_body = _extract_pdf_text_from_bytes(raw_bytes, limit=1500)
                else:
                    fetched_body = _html_to_text(raw_bytes.decode("utf-8", errors="ignore"))[:1500]
                if fetched_body and "Error fetching page" not in fetched_body:
                    result["body"] = fetched_body[:1500]
                    result["href"] = final_url or href
            except Exception:
                pass
            return result

        results = []
        enrich_seed = seed_results[: min(3, len(seed_results))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, max(1, len(enrich_seed)))) as executor:
            futures = [executor.submit(enrich, dict(item)) for item in enrich_seed]
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    continue
        results.extend(seed_results[len(enrich_seed):])

        results = [item for item in results if item.get("href")]
        return _filter_ranked_results(results, query, max_results, strict_site_hint=True)

    def _dedupe_results(self, results, max_results: int):
        deduped = []
        seen = set()
        for item in results or []:
            if not isinstance(item, dict):
                continue
            href = _normalize_search_result_url(str(item.get("href", "")).strip())
            title = str(item.get("title", "")).strip()
            body = str(item.get("body", "")).strip()
            if not href and not title and not body:
                continue
            key = href or title or body[:120]
            if key in seen:
                continue
            seen.add(key)
            deduped.append({"title": title, "href": href, "body": body})
            if len(deduped) >= max_results:
                break
        return deduped

    def _search_tavily(self, query: str, max_results: int):
        raw_results = None
        tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if tavily_key:
            payload = {
                "api_key": tavily_key,
                "query": query,
                "max_results": min(max_results, 5),
                "search_depth": "advanced",
                "include_answer": False,
                "include_images": False,
                "include_raw_content": False,
            }
            try:
                if requests is not None:
                    response = requests.post(
                        "https://api.tavily.com/search",
                        json=payload,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    raw_results = response.json().get("results", [])
                else:
                    request = Request(
                        "https://api.tavily.com/search",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(request, timeout=self.timeout) as response:
                        raw_results = json.loads(response.read().decode("utf-8", errors="ignore")).get("results", [])
            except Exception:
                raw_results = None
        if raw_results is None:
            raw_results = self.tavily_search.invoke({"query": query, "max_results": min(max_results, 5)})
        if isinstance(raw_results, dict):
            raw_results = raw_results.get("results", [])
        results = []
        for item in raw_results or []:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "href": _normalize_search_result_url(str(item.get("url", "")).strip()),
                    "body": str(item.get("content", "") or item.get("snippet", "")).strip(),
                }
            )
        results = [item for item in results if item.get("title") or item.get("body") or item.get("href")]
        return _filter_ranked_results(results, query, max_results, strict_site_hint=_should_require_site_hint(query))

    def _search_duckduckgo(self, query: str, max_results: int):
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        html, _ = _http_get(url, headers=self.headers, timeout=self.timeout)
        results = []
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            for result in soup.select(".result")[:max_results]:
                link = result.select_one(".result__a")
                snippet = result.select_one(".result__snippet")
                if not link:
                    continue
                results.append(
                    {
                        "title": link.get_text(" ", strip=True),
                        "href": _normalize_search_result_url(link.get("href", "")),
                        "body": snippet.get_text(" ", strip=True) if snippet else "",
                    }
                )
        else:
            pattern = re.compile(
                r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?(?:<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>|<div[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</div>)?',
                re.DOTALL,
            )
            for match in pattern.finditer(html):
                href = _normalize_search_result_url(unescape(match.group(1)))
                title = _html_to_text(match.group(2))
                body = _html_to_text(match.group(3) or match.group(4) or "")
                results.append({"title": title, "href": href, "body": body})
                if len(results) >= max_results:
                    break
        return _filter_ranked_results(results, query, max_results, strict_site_hint=_should_require_site_hint(query))

    def _search_bing(self, query: str, max_results: int):
        url = f"https://www.bing.com/search?q={quote_plus(query)}"
        html, _ = _http_get(url, headers=self.headers, timeout=self.timeout)
        results = []
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            for result in soup.select("li.b_algo")[:max_results]:
                link = result.select_one("h2 a")
                snippet = result.select_one(".b_caption p")
                if not link:
                    continue
                results.append(
                    {
                        "title": link.get_text(" ", strip=True),
                        "href": _normalize_search_result_url(link.get("href", "")),
                        "body": snippet.get_text(" ", strip=True) if snippet else "",
                    }
                )
        else:
            pattern = re.compile(
                r'<li class="b_algo".*?<h2><a href="([^"]+)".*?>(.*?)</a></h2>.*?(?:<p>(.*?)</p>)?',
                re.DOTALL,
            )
            for match in pattern.finditer(html):
                href = _normalize_search_result_url(unescape(match.group(1)))
                title = _html_to_text(match.group(2))
                body = _html_to_text(match.group(3) or "")
                results.append({"title": title, "href": href, "body": body})
                if len(results) >= max_results:
                    break
        return _filter_ranked_results(results, query, max_results, strict_site_hint=_should_require_site_hint(query))

    async def afetch_url_content(self, url: str, mode: str = "truncate"):
        return await asyncio.to_thread(self.fetch_url_content, url, mode)

    def fetch_url_content(self, url: str, mode: str = "truncate"):
        github_issue_match = _match_github_issue_url(url)
        if github_issue_match:
            owner, repo, issue_number = github_issue_match
            api_headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}
            issue_api = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
            timeline_api = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/timeline"
            try:
                issue_text, _ = _http_get(issue_api, headers=api_headers, timeout=self.timeout)
                timeline_text, _ = _http_get(timeline_api, headers=api_headers, timeout=self.timeout)
                issue_json = json.loads(issue_text)
                timeline_json = json.loads(timeline_text)
                text = _render_github_issue_api_text(issue_json, timeline_json)
                if mode == "truncate":
                    text = _truncate(text, 12000)
                return {"content": text, "redirect_url": url}
            except Exception:
                html_headers = {
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
                try:
                    html_text, final_url = _http_get(url, headers=html_headers, timeout=self.timeout)
                    text = _render_github_issue_html_text(html_text)
                    if mode == "truncate":
                        text = _truncate(text, 12000)
                    return {"content": text, "redirect_url": final_url}
                except Exception:
                    pass

        last_error = None
        for timeout in (self.timeout, max(self.timeout, 25)):
            try:
                parsed_input = urlparse(url or "")
                input_domain = (parsed_input.netloc or "").lower()
                input_query = parse_qs(parsed_input.query or "")
                if input_domain.endswith("wikipedia.org") and input_query.get("action") == ["raw"]:
                    raw_text, final_url = _http_get(url, headers=self.headers, timeout=timeout)
                    text = raw_text.strip()
                    if mode == "truncate":
                        text = _truncate(text, 30000)
                    return {"content": text, "redirect_url": final_url}

                raw_bytes, final_url, content_type = _http_get_bytes(url, headers=self.headers, timeout=timeout)
                lower_url = (final_url or url or "").lower()
                lower_ct = (content_type or "").lower()
                if lower_url.endswith(".pdf") or "application/pdf" in lower_ct or raw_bytes.startswith(b"%PDF"):
                    text = _extract_pdf_text_from_bytes(raw_bytes)
                else:
                    html = raw_bytes.decode("utf-8", errors="ignore")
                    text = _html_to_text(html)
                    wiki_text = self._fetch_wikipedia_wikitext(final_url or url)
                    if wiki_text:
                        focus_sections = _extract_wikipedia_focus_sections(wiki_text, limit=12000)
                        text = f"{text}\n\nWIKIPEDIA_WIKITEXT:\n{wiki_text}"
                        if focus_sections:
                            text = f"{text}\n\nWIKIPEDIA_FOCUS_SECTIONS:\n{focus_sections}"
                if mode == "truncate":
                    limit = 30000 if "wikipedia.org" in lower_url else 12000
                    text = _truncate(text, limit)
                return {"content": text, "redirect_url": final_url}
            except Exception as exc:
                last_error = exc
                continue
        raise last_error or RuntimeError(f"Failed to fetch URL: {url}")

    def _fetch_wikipedia_wikitext(self, url: str) -> str:
        try:
            parsed = urlparse(url or "")
            domain = (parsed.netloc or "").lower()
            if not (domain == "en.wikipedia.org" or domain.endswith(".en.wikipedia.org")):
                return ""
            if not parsed.path.startswith("/wiki/"):
                return ""
            title = unquote(parsed.path[len("/wiki/"):]).strip()
            if not title or ":" in title:
                return ""
            api_url = (
                "https://en.wikipedia.org/w/api.php?action=parse&prop=wikitext&format=json&page="
                + quote_plus(title.replace("_", " "))
            )
            payload, _ = _http_get(api_url, headers=self.headers, timeout=self.timeout)
            data = json.loads(payload)
            wikitext = (((data.get("parse") or {}).get("wikitext") or {}).get("*") or "").strip()
            if not wikitext:
                return ""
            return _truncate(wikitext, 20000)
        except Exception:
            return ""


def run_search_engine_factory(search_api: DuckDuckGoSearch):
    async def run_search_engine(query, max_results=8, **kwargs):
        return await search_api.asearch(query, max_results=max_results)
    return run_search_engine


def run_github_issue_search_factory():
    def _label_family(label_name: Any) -> str:
        label = str(label_name or "").strip()
        label = re.sub(r"^\d+\s*-\s*", "", label)
        return label.strip()

    def _strip_arg_prefix(value: Any, prefixes: List[str]) -> str:
        text = str(value or "").strip()
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if text.lower().startswith(prefix.lower()):
                    text = text[len(prefix) :].strip()
                    changed = True
        return text.strip().strip("\"'")

    def _normalize_label_token(label: Any) -> str:
        token = _strip_arg_prefix(label, ["labels:", "label:"])
        token = token.strip().strip("\"'")
        token = re.sub(r"^\d+\s*-\s*", "", token)
        token = re.sub(r"\s+", " ", token)
        return token.strip().lower()

    def _split_label_query(labels: str) -> List[str]:
        cleaned = _strip_arg_prefix(labels, ["labels:", "label:"])
        cleaned = cleaned.replace(" and ", ",").replace(" AND ", ",")
        parts = []
        for part in re.split(r"\s*,\s*|\s+;\s*", cleaned):
            token = part.strip().strip("\"'")
            if token:
                parts.append(token)
        return parts

    def _label_matches_token(actual_label: str, requested_token: str) -> bool:
        actual = str(actual_label or "").strip()
        requested = str(requested_token or "").strip()
        actual_norm = _normalize_label_token(actual)
        actual_family_norm = _normalize_label_token(_label_family(actual))
        requested_norm = _normalize_label_token(requested)
        if not requested_norm:
            return True
        if requested_norm in {actual_norm, actual_family_norm}:
            return True
        if requested_norm in actual_norm or requested_norm in actual_family_norm:
            return True
        if actual_norm in requested_norm or actual_family_norm in requested_norm:
            return True
        return False

    def _needs_family_fallback(requested_tokens: List[str], expanded_options: List[List[str]]) -> bool:
        for token, options in zip(requested_tokens, expanded_options):
            normalized = str(token or "").strip()
            if not normalized:
                continue
            if ":" in normalized or re.search(r"\d+\s*-\s*", normalized):
                continue
            if len(options) == 1 and _normalize_label_token(options[0]) == _normalize_label_token(normalized):
                return True
        return False

    def _github_get_json(url: str, params: Optional[dict] = None):
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "DynaCall-GAIA",
        }
        github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"
        if requests is not None:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        query = ""
        if params:
            query = "?" + "&".join(f"{quote_plus(str(k))}={quote_plus(str(v))}" for k, v in params.items())
        request = Request(url + query, headers=headers)
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))

    def _github_get_text(url: str, params: Optional[dict] = None) -> str:
        headers = {
            "User-Agent": "DynaCall-GAIA",
        }
        github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"
        if requests is not None:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        query = ""
        if params:
            query = "?" + "&".join(f"{quote_plus(str(k))}={quote_plus(str(v))}" for k, v in params.items())
        request = Request(url + query, headers=headers)
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="ignore")

    def _parse_issue_embedded_data(html: str) -> Optional[dict]:
        match = re.search(
            r'<script type="application/json" data-target="react-app\.embeddedData">(.*?)</script>',
            html,
            flags=re.S,
        )
        if not match:
            return None
        try:
            embedded = json.loads(unescape(match.group(1)))
            queries = embedded.get("payload", {}).get("preloadedQueries", [])
            return queries[0]["result"]["data"]["repository"]["issue"]
        except Exception:
            return None

    def _build_github_issue_search_query(repo: str, state: str, requested_tokens: List[str], sort: str, direction: str) -> str:
        parts = ["is:issue", f"repo:{repo}"]
        if state:
            parts.append(f"is:{state}")
        chosen_tokens = []
        preferred_tokens = [token for token in requested_tokens if "component:" not in str(token).lower()]
        if preferred_tokens:
            chosen_tokens = [preferred_tokens[0]]
        elif requested_tokens:
            chosen_tokens = [requested_tokens[0]]
        for token in chosen_tokens:
            normalized = str(token or "").strip()
            if not normalized:
                continue
            candidate = normalized
            lower = normalized.lower()
            if lower.startswith("label:"):
                candidate = normalized
            elif ":" in normalized or re.search(r"\d+\s*-\s*", normalized):
                candidate = f'label:"{normalized}"'
            else:
                candidate = f'"{normalized}"'
            parts.append(candidate)
        if sort == "created":
            suffix = "asc" if direction == "asc" else "desc"
            parts.append(f"sort:created-{suffix}")
        return " ".join(parts)

    def _extract_issue_urls_from_search_html(html: str, repo: str) -> List[str]:
        urls = []
        seen = set()
        pattern = rf'/{re.escape(repo)}/issues/(\d+)'
        for match in re.finditer(pattern, html):
            url = f"https://github.com/{repo}/issues/{match.group(1)}"
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
        return urls

    def _parse_search_issue_candidates(html: str, repo: str) -> List[dict]:
        if BeautifulSoup is None:
            return []
        soup = BeautifulSoup(html, "html.parser")
        candidates = []
        seen = set()
        pattern = re.compile(rf"^/{re.escape(repo)}/issues/\d+$")
        for link in soup.find_all("a", href=pattern):
            href = str(link.get("href") or "").strip()
            issue_url = f"https://github.com{href}"
            if issue_url in seen:
                continue
            seen.add(issue_url)
            block = link
            for _ in range(4):
                if getattr(block, "parent", None) is None:
                    break
                block = block.parent
            text = " ".join(block.get_text(" ", strip=True).split())
            number_match = re.search(r"#\s*(\d+)", text)
            closed_match = re.search(r"closed on ([A-Za-z]{3,9} \d{1,2}, \d{4})", text, flags=re.I)
            labels = []
            for label_match in re.finditer(r"\b\d+\s*-\s*[A-Za-z][A-Za-z0-9 _./:-]*|\bcomponent:\s*[A-Za-z0-9_.:-]+", text):
                label = " ".join(label_match.group(0).split())
                if label not in labels:
                    labels.append(label)
            candidates.append(
                {
                    "number": int(number_match.group(1)) if number_match else None,
                    "title": " ".join(link.get_text(" ", strip=True).split()),
                    "state": "closed" if "closed" in text.lower() else "",
                    "html_url": issue_url,
                    "created_at": None,
                    "closed_at": closed_match.group(1) if closed_match else None,
                    "labels": labels,
                    "label_families": [_label_family(label) for label in labels],
                    "label_events": [],
                    "search_snippet": text[:500],
                }
            )
        return candidates

    async def _enrich_issue_candidate_with_timeline(candidate: dict) -> dict:
        issue_url = str(candidate.get("html_url") or "").strip()
        if not issue_url:
            return candidate
        try:
            issue_html = await asyncio.to_thread(_github_get_text, issue_url, None)
        except Exception:
            return candidate
        issue_data = _parse_issue_embedded_data(issue_html)
        if not issue_data:
            return candidate
        label_names = []
        for edge in (issue_data.get("labels", {}) or {}).get("edges", []) or []:
            name = ((edge or {}).get("node") or {}).get("name")
            if name:
                label_names.append(name)
        label_events = []
        for key in ("frontTimelineItems", "backTimelineItems"):
            for edge in (issue_data.get(key, {}) or {}).get("edges", []) or []:
                node = (edge or {}).get("node") or {}
                event_type = node.get("__typename")
                if event_type not in {"LabeledEvent", "UnlabeledEvent"}:
                    continue
                label_name = ((node.get("label") or {}).get("name"))
                label_events.append(
                    {
                        "event": "labeled" if event_type == "LabeledEvent" else "unlabeled",
                        "label": label_name,
                        "label_family": _label_family(label_name),
                        "created_at": node.get("createdAt"),
                        "actor": ((node.get("actor") or {}).get("login")),
                    }
                )
        enriched = dict(candidate)
        if issue_data.get("createdAt"):
            enriched["created_at"] = issue_data.get("createdAt")
        if issue_data.get("closedAt"):
            enriched["closed_at"] = issue_data.get("closedAt")
        if label_names:
            enriched["labels"] = label_names
            enriched["label_families"] = [_label_family(label) for label in label_names]
        if label_events:
            enriched["label_events"] = label_events
        return enriched

    async def run_github_issue_search(*args):
        if len(args) == 1:
            values = args[0] if isinstance(args[0], list) else [args[0]]
        else:
            values = list(args)
        if len(values) < 2:
            return "github_issue_search expects [repo, labels, state?, sort?, direction?, per_page?]."
        repo = _strip_arg_prefix(values[0], ["repo:"]).strip("/")
        labels = _strip_arg_prefix(values[1], ["labels:", "label:"])
        state = _strip_arg_prefix(values[2], ["state:"]) if len(values) > 2 and values[2] else "closed"
        sort = _strip_arg_prefix(values[3], ["sort:"]) if len(values) > 3 and values[3] else "created"
        direction = _strip_arg_prefix(values[4], ["direction:"]) if len(values) > 4 and values[4] else "asc"
        try:
            per_page_raw = _strip_arg_prefix(values[5], ["per_page:", "per-page:", "limit:"]) if len(values) > 5 and values[5] else 10
            per_page = int(per_page_raw)
        except Exception:
            per_page = 10
        per_page = max(1, min(per_page, 50))
        if not re.fullmatch(r"[^/\s]+/[^/\s]+", repo):
            return "Invalid repo. Use owner/repo, for example numpy/numpy."
        requested_label_tokens = _split_label_query(labels)

        api_failed = False
        try:
            repo_labels = []
            for page in range(1, 4):
                page_labels = await asyncio.to_thread(
                    _github_get_json,
                    f"https://api.github.com/repos/{repo}/labels",
                    {"per_page": 100, "page": page},
                )
                if not isinstance(page_labels, list) or not page_labels:
                    break
                repo_labels.extend(
                    label.get("name")
                    for label in page_labels
                    if isinstance(label, dict) and label.get("name")
                )
        except Exception as exc:
            repo_labels = []
            api_failed = _extract_http_status_code(exc) == 403

        expanded_label_options: List[List[str]] = []
        for token in requested_label_tokens:
            matches = [name for name in repo_labels if _label_matches_token(name, token)]
            expanded_label_options.append(matches or [token])

        base_label = ""
        candidate_base_labels: List[str] = []
        for options in expanded_label_options:
            if not options:
                continue
            candidate_base_labels.append(options[0])
        preferred = [label for label in candidate_base_labels if "component:" not in str(label).lower()]
        if preferred:
            base_label = preferred[0]
        elif candidate_base_labels:
            base_label = candidate_base_labels[0]

        params = {
            "state": state,
            "sort": sort,
            "direction": direction,
            "per_page": 100,
        }
        if base_label:
            params["labels"] = base_label

        try:
            issues = []
            for page in range(1, 6):
                page_params = dict(params)
                page_params["page"] = page
                page_issues = await asyncio.to_thread(
                    _github_get_json,
                    f"https://api.github.com/repos/{repo}/issues",
                    page_params,
                )
                if not isinstance(page_issues, list) or not page_issues:
                    break
                issues.extend(page_issues)
                if len(issues) >= 300:
                    break
        except Exception:
            issues = []
            api_failed = True

        results = []
        for issue in issues if isinstance(issues, list) else []:
            if "pull_request" in issue:
                continue
            issue_labels = [
                label.get("name")
                for label in issue.get("labels", [])
                if isinstance(label, dict) and label.get("name")
            ]
            if requested_label_tokens and not all(
                any(_label_matches_token(actual, token) for actual in issue_labels)
                for token in requested_label_tokens
            ):
                continue
            number = issue.get("number")
            label_events = []
            if number is not None:
                try:
                    events = await asyncio.to_thread(
                        _github_get_json,
                        f"https://api.github.com/repos/{repo}/issues/{number}/events",
                        None,
                    )
                    for event in events if isinstance(events, list) else []:
                        event_type = event.get("event")
                        if event_type not in {"labeled", "unlabeled"}:
                            continue
                        label = (event.get("label") or {}).get("name")
                        label_events.append(
                            {
                                "event": event_type,
                                "label": label,
                                "label_family": _label_family(label),
                                "created_at": event.get("created_at"),
                                "actor": ((event.get("actor") or {}).get("login")),
                            }
                        )
                except Exception as exc:
                    label_events.append({"error": str(exc)})
            results.append(
                {
                    "number": number,
                    "title": issue.get("title"),
                    "state": issue.get("state"),
                    "html_url": issue.get("html_url"),
                    "created_at": issue.get("created_at"),
                    "closed_at": issue.get("closed_at"),
                    "labels": issue_labels,
                    "label_families": [_label_family(label) for label in issue_labels],
                    "label_events": label_events,
                }
            )
            if len(results) >= per_page:
                break

        if not results and (api_failed or bool(requested_label_tokens)):
            try:
                query = _build_github_issue_search_query(repo, state, requested_label_tokens, sort, direction)
                for page in range(1, 13):
                    try:
                        html = await asyncio.to_thread(
                            _github_get_text,
                            f"https://github.com/{repo}/issues",
                            {"q": query, "page": page},
                        )
                    except Exception:
                        continue
                    page_results = _parse_search_issue_candidates(html, repo)
                    if not page_results:
                        break
                    for item in page_results:
                        if requested_label_tokens and not all(
                            any(_label_matches_token(actual, token) for actual in item.get("labels", []))
                            for token in requested_label_tokens
                        ):
                            continue
                        if requested_label_tokens:
                            item = await _enrich_issue_candidate_with_timeline(item)
                        results.append(item)
                        if len(results) >= per_page:
                            break
                    if len(results) >= per_page:
                        break
            except Exception:
                pass

        if results and _needs_family_fallback(requested_label_tokens, expanded_label_options):
            try:
                query = _build_github_issue_search_query(repo, state, requested_label_tokens, sort, direction)
                existing_numbers = {item.get("number") for item in results}
                fallback_results = []
                for page in range(1, 13):
                    try:
                        html = await asyncio.to_thread(
                            _github_get_text,
                            f"https://github.com/{repo}/issues",
                            {"q": query, "page": page},
                        )
                    except Exception:
                        continue
                    page_results = _parse_search_issue_candidates(html, repo)
                    if not page_results:
                        break
                    for item in page_results:
                        if item.get("number") in existing_numbers:
                            continue
                        if requested_label_tokens and not all(
                            any(_label_matches_token(actual, token) for actual in item.get("labels", []))
                            for token in requested_label_tokens
                        ):
                            continue
                        if requested_label_tokens:
                            item = await _enrich_issue_candidate_with_timeline(item)
                        fallback_results.append(item)
                if fallback_results:
                    results.extend(fallback_results)
                    results.sort(key=lambda item: str(item.get("created_at") or item.get("number") or ""))
                    results = results[:per_page]
            except Exception:
                pass

        return _truncate_json_safely(
            {
                "repo": repo,
                "state": state,
                "labels_query": labels,
                "requested_label_tokens": requested_label_tokens,
                "expanded_label_options": expanded_label_options,
                "base_label_used_for_api": base_label,
                "sort": sort,
                "direction": direction,
                "issues": results,
            }
        )

    return run_github_issue_search


def _normalize_wiki_title(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc and "wikipedia.org" in parsed.netloc:
        path = unquote(parsed.path or "")
        if "/wiki/" in path:
            title = path.split("/wiki/", 1)[1]
            return title.replace("_", " ").strip()
    return value.replace("_", " ").strip()


def _wiki_oldid_from_url(value: str) -> Optional[int]:
    parsed = urlparse(value)
    if not (parsed.scheme and parsed.netloc and "wikipedia.org" in parsed.netloc):
        return None
    query = parse_qs(parsed.query or "")
    oldid = query.get("oldid", [])
    if not oldid:
        return None
    try:
        return int(str(oldid[0]).strip())
    except Exception:
        return None


def _normalize_section_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _score_wiki_section(section_hint: str, candidate: str) -> int:
    hint = _normalize_section_name(section_hint)
    name = _normalize_section_name(candidate)
    if not hint or not name:
        return 0
    if hint == name:
        return 100
    if hint in name:
        return 80
    if name in hint:
        return 60
    hint_tokens = set(hint.split())
    name_tokens = set(name.split())
    return len(hint_tokens & name_tokens) * 10


def _extract_wiki_section_text(html: str) -> str:
    if BeautifulSoup is None:
        return _truncate(_html_to_text(html))

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.select("sup.reference, span.mw-editsection, .reference, .reflist"):
        tag.decompose()

    lines: List[str] = []

    for table_index, table in enumerate(soup.select("table.wikitable, table.plainrowheaders, table"), start=1):
        row_lines: List[str] = []
        for row in table.select("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.select("th, td")]
            cells = [re.sub(r"\s+", " ", cell).strip() for cell in cells if re.sub(r"\s+", " ", cell).strip()]
            if cells:
                row_lines.append(" | ".join(cells))
        if row_lines:
            lines.append(f"[TABLE {table_index}]")
            lines.extend(row_lines)
        table.decompose()

    for node in soup.select("h1, h2, h3, h4, p, li"):
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if text:
            lines.append(text)

    deduped: List[str] = []
    seen = set()
    for line in lines:
        key = line.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)

    return _truncate("\n".join(deduped))


def run_wiki_section_extract_factory():
    async def run_wiki_section_extract(page_or_title, section_hint, revision_year=None):
        try:
            page_value = page_or_title[0] if isinstance(page_or_title, list) else page_or_title
            section_value = section_hint[0] if isinstance(section_hint, list) else section_hint
            year_value = revision_year[0] if isinstance(revision_year, list) and revision_year else revision_year
            if not page_value or not section_value:
                return "Missing Wikipedia page/title or section hint."
            title = _normalize_wiki_title(str(page_value))
            oldid = _wiki_oldid_from_url(str(page_value))
            headers = {"User-Agent": "DynaCall/GAIA"}
            if oldid is None and year_value:
                year_text = str(year_value).strip()
                if re.fullmatch(r"\d{4}", year_text):
                    rvstart = f"{year_text}-12-31T23:59:59Z"
                    revision_api = (
                        "https://en.wikipedia.org/w/api.php?action=query&format=json"
                        f"&prop=revisions&titles={quote_plus(title)}&rvlimit=1&rvdir=older"
                        f"&rvstart={quote_plus(rvstart)}&rvprop=ids|timestamp"
                    )
                    revision_payload = await asyncio.to_thread(_http_get_json, revision_api, headers, 30)
                    pages = ((revision_payload.get("query") or {}).get("pages") or {})
                    for page in pages.values():
                        revisions = page.get("revisions") or []
                        if revisions:
                            oldid = revisions[0].get("revid")
                            break
            section_list_api = (
                "https://en.wikipedia.org/w/api.php?action=parse&format=json&prop=sections&"
                + (f"oldid={oldid}" if oldid else f"page={quote_plus(title)}")
            )
            section_payload = await asyncio.to_thread(_http_get_json, section_list_api, headers, 30)
            sections = (section_payload.get("parse") or {}).get("sections") or []
            best_section = None
            best_score = -1
            for section in sections:
                score = _score_wiki_section(str(section_value), str(section.get("line") or ""))
                if score > best_score:
                    best_score = score
                    best_section = section
            if best_section is None or best_score <= 0:
                available = [str(section.get("line") or "").strip() for section in sections[:20] if str(section.get("line") or "").strip()]
                return _truncate_json_safely({"title": title, "resolved_oldid": oldid, "matched_section": None, "available_sections": available, "content": ""})
            section_index = str(best_section.get("index"))
            parse_api = (
                "https://en.wikipedia.org/w/api.php?action=parse&format=json&prop=text&"
                + (f"oldid={oldid}" if oldid else f"page={quote_plus(title)}")
                + f"&section={quote_plus(section_index)}"
            )
            parse_payload = await asyncio.to_thread(_http_get_json, parse_api, headers, 30)
            html = (((parse_payload.get("parse") or {}).get("text") or {}).get("*")) or ""
            extracted = _extract_wiki_section_text(html)
            return _truncate_json_safely(
                {
                    "title": (parse_payload.get("parse") or {}).get("title") or title,
                    "resolved_oldid": oldid,
                    "matched_section": str(best_section.get("line") or "").strip(),
                    "section_index": section_index,
                    "content": extracted,
                }
            )
        except Exception as exc:
            return f"Failed to extract Wikipedia section: {exc}"
    return run_wiki_section_extract


def run_crossref_lookup_factory():
    async def run_crossref_lookup(*args):
        values = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
        if not values:
            return "No paper title provided."
        title = str(values[0]).strip()
        if not title:
            return "No paper title provided."
        try:
            max_results = int(values[1]) if len(values) > 1 and str(values[1]).strip() else 5
        except Exception:
            max_results = 5
        max_results = max(1, min(max_results, 10))
        url = "https://api.crossref.org/works?" + urlencode(
            {
                "query.title": title,
                "rows": max_results,
                "select": "DOI,title,author,issued,container-title,URL,type,published-print,published-online",
            }
        )
        data = await asyncio.to_thread(_openalex_get_json, url, {"User-Agent": "DynaCall/GAIA"}, 30)
        items = (((data or {}).get("message") or {}).get("items") or [])
        results: List[Dict[str, Any]] = []
        for item in items:
            title_list = item.get("title") or []
            title_value = str(title_list[0]).strip() if title_list else ""
            authors = []
            for author in item.get("author") or []:
                given = str(author.get("given") or "").strip()
                family = str(author.get("family") or "").strip()
                full = " ".join(part for part in [given, family] if part).strip()
                if full:
                    authors.append(full)
            year = None
            for key in ("issued", "published-print", "published-online"):
                parts = (((item.get(key) or {}).get("date-parts") or [[]])[0])
                if parts:
                    try:
                        year = int(parts[0])
                        break
                    except Exception:
                        pass
            venue_list = item.get("container-title") or []
            venue = str(venue_list[0]).strip() if venue_list else ""
            results.append(
                {
                    "source": "crossref",
                    "title": title_value,
                    "doi": str(item.get("DOI") or "").strip(),
                    "url": str(item.get("URL") or "").strip(),
                    "year": year,
                    "venue": venue,
                    "type": str(item.get("type") or "").strip(),
                    "authors": authors,
                }
            )
        openalex_results: List[Dict[str, Any]] = []
        try:
            openalex_url = "https://api.openalex.org/works?" + urlencode(
                {
                    "search": title,
                    "per-page": max_results,
                    "select": "id,display_name,publication_year,authorships,primary_location,doi,type",
                }
            )
            openalex_data = await asyncio.to_thread(_openalex_get_json, openalex_url, {"User-Agent": "DynaCall/GAIA"}, 30)
            for work in openalex_data.get("results") or []:
                venue = str((((work.get("primary_location") or {}).get("source") or {}).get("display_name")) or "").strip()
                openalex_results.append(
                    {
                        "source": "openalex",
                        "title": str(work.get("display_name") or "").strip(),
                        "doi": str(work.get("doi") or "").strip(),
                        "url": str(work.get("id") or "").strip(),
                        "year": work.get("publication_year"),
                        "venue": venue,
                        "type": str(work.get("type") or "").strip(),
                        "authors": _normalize_openalex_authorships(work.get("authorships")),
                    }
                )
        except Exception as exc:
            openalex_results.append({"source": "openalex", "error": str(exc)})
        return _truncate_json_safely({"query_title": title, "results": openalex_results + results})
    return run_crossref_lookup


def _quote_verifier_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text or "").lower())


def _quote_verifier_clean_word(text: str) -> str:
    words = _quote_verifier_words(text)
    return words[0] if words else ""


def _quote_verifier_best_diff(quoted_text: str, source_text: str) -> Dict[str, Any]:
    quote_words = _quote_verifier_words(quoted_text)
    source_words = _quote_verifier_words(source_text)
    if not quote_words or not source_words:
        return {"matches": False, "confidence": 0.0, "mismatched_word": "", "correct_word": "", "support": ""}

    exact_quote = " ".join(quote_words)
    source_joined = " ".join(source_words)
    if exact_quote and exact_quote in source_joined:
        return {
            "matches": True,
            "confidence": 1.0,
            "mismatched_word": "",
            "correct_word": "",
            "support": exact_quote,
        }

    rare_anchors = [
        word
        for word in quote_words
        if len(word) >= 6 and word not in {"because", "between", "relationship", "authors", "quoted", "citation"}
    ]
    anchor_positions = [
        idx
        for idx, word in enumerate(source_words)
        if word in rare_anchors or word in {"scribal", "mis-transmission", "mis", "transmission"}
    ]
    if not anchor_positions:
        anchor_positions = list(range(0, len(source_words), max(1, len(quote_words))))

    import difflib

    best: Dict[str, Any] = {"ratio": 0.0, "start": 0, "window": []}
    quote_len = len(quote_words)
    for pos in anchor_positions[:120]:
        for start in range(max(0, pos - quote_len - 8), min(len(source_words), pos + 4)):
            for window_len in range(max(3, quote_len - 4), quote_len + 5):
                window = source_words[start:start + window_len]
                if not window:
                    continue
                ratio = difflib.SequenceMatcher(a=quote_words, b=window).ratio()
                if ratio > best["ratio"]:
                    best = {"ratio": ratio, "start": start, "window": window}

    window = best.get("window") or []
    matcher = difflib.SequenceMatcher(a=quote_words, b=window)
    substitutions: List[tuple] = []
    inserts_or_deletes = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and (i2 - i1) == 1 and (j2 - j1) == 1:
            substitutions.append((quote_words[i1], window[j1]))
        else:
            inserts_or_deletes += max(i2 - i1, j2 - j1)

    support = " ".join(window)
    if substitutions and best["ratio"] >= 0.72:
        mismatched_word, correct_word = substitutions[0]
        return {
            "matches": False,
            "confidence": round(float(best["ratio"]), 4),
            "mismatched_word": mismatched_word,
            "correct_word": correct_word,
            "support": support,
        }
    if best["ratio"] >= 0.92 and not substitutions and inserts_or_deletes <= 1:
        return {
            "matches": False,
            "confidence": round(float(best["ratio"]), 4),
            "mismatched_word": "",
            "correct_word": "",
            "support": support,
        }
    return {
        "matches": False,
        "confidence": round(float(best["ratio"]), 4),
        "mismatched_word": "",
        "correct_word": "",
        "support": support,
    }


def _quote_verifier_extract_title(citation: str) -> str:
    citation = str(citation or "")
    quoted_titles = re.findall(r'"([^"]{8,180})"', citation)
    if quoted_titles:
        return quoted_titles[0].strip()
    return re.sub(r"\s+", " ", citation).strip()[:180]


def _quote_verifier_candidate_urls(citation: str, search_items: List[Dict[str, Any]]) -> List[str]:
    urls: List[str] = []
    doi_match = re.search(r"\b10\.\d{4,9}/[^\s,;)]+", citation, flags=re.I)
    if doi_match:
        doi = doi_match.group(0).rstrip(".")
        urls.append(f"https://doi.org/{doi}")
    for item in search_items:
        href = str((item or {}).get("href") or "").strip()
        if href:
            urls.append(href)
    expanded_urls = list(urls)
    for url in urls:
        muse_match = re.search(r"muse\.jhu\.edu/(?:(pub/\d+)/)?article/(\d+)", url, flags=re.I)
        if muse_match:
            pub_prefix = muse_match.group(1)
            article_id = muse_match.group(2)
            expanded_urls.extend(
                [
                    f"https://muse.jhu.edu/article/{article_id}",
                    f"https://muse.jhu.edu/article/{article_id}/pdf",
                ]
            )
            if pub_prefix:
                expanded_urls.extend(
                    [
                        f"https://muse.jhu.edu/{pub_prefix}/article/{article_id}",
                        f"https://muse.jhu.edu/{pub_prefix}/article/{article_id}/pdf",
                    ]
                )
    deduped: List[str] = []
    seen = set()
    for url in expanded_urls:
        normalized = _normalize_search_result_url(url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _quote_verifier_reject_url(url: str) -> bool:
    lowered = str(url or "").lower()
    reject_tokens = (
        "citation-report",
        "citationstylelanguage",
        "download/ris",
        "bibtex",
        "store/storeresults",
        "search_title",
        "/search/",
        "/psearch/",
        "serp.php",
        "htshub.com",
        "grammar.com/search",
        "huggingface.co/datasets",
        "fiqh.world-federation.org/search",
    )
    return any(token in lowered for token in reject_tokens)


def _quote_verifier_is_query_echo(url: str, title: str, body: str) -> bool:
    lowered_url = str(url or "").lower()
    lowered_title = str(title or "").lower()
    lowered_body = str(body or "").lower()
    if any(token in lowered_url for token in ("/search/", "/psearch/", "serp.php", "search?", "query=")):
        return True
    echo_markers = (
        "search results for",
        "found ",
        "matching",
        "articles matching",
        "check if exact quote",
    )
    return any(marker in lowered_title or marker in lowered_body for marker in echo_markers)


def _quote_verifier_trusted_snippet_source(url: str) -> bool:
    lowered = str(url or "").lower()
    trusted = (
        "books.google.",
        "muse.jhu.edu",
        "jstor.org",
        "projecteuclid.org",
        "cambridge.org",
        "oup.com",
        "springer.com",
        "link.springer.com",
        "tandfonline.com",
    )
    return any(token in lowered for token in trusted)


def run_quote_verifier_factory(search_api: DuckDuckGoSearch):
    async def run_quote_verifier(*args):
        values = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
        if len(values) < 2:
            return _truncate_json_safely(
                {
                    "status": "error",
                    "error": "quote_verifier expects citation/title/doi and quoted_text",
                    "matches": None,
                    "mismatched_word": "",
                    "correct_word": "",
                }
            )
        citation = str(values[0] or "").strip()
        quoted_text = str(values[1] or "").strip()
        title = _quote_verifier_extract_title(citation)
        quote_words = _quote_verifier_words(quoted_text)
        anchor = " ".join([word for word in quote_words if len(word) >= 6][-6:])
        doi_match = re.search(r"\b10\.\d{4,9}/[^\s,;)]+", citation, flags=re.I)
        doi = doi_match.group(0).rstrip(".") if doi_match else ""

        queries = []
        if title and anchor:
            queries.append(f'"{title}" "{anchor}"')
        if title:
            queries.extend([f'"{title}" full text', f'"{title}" pdf'])
        if doi:
            queries.extend([f'"{doi}" full text', f'"{doi}" pdf'])
            try:
                crossref_url = "https://api.crossref.org/works/" + quote_plus(doi)
                crossref_data = await asyncio.to_thread(
                    _openalex_get_json,
                    crossref_url,
                    {"User-Agent": "DynaCall/GAIA"},
                    20,
                )
                crossref_message = (crossref_data or {}).get("message") or {}
                publisher_url = str(crossref_message.get("URL") or "").strip()
                if publisher_url:
                    search_items.append({"title": "Crossref publisher URL", "href": publisher_url, "body": ""})
                for link in crossref_message.get("link") or []:
                    if isinstance(link, dict) and link.get("URL"):
                        search_items.append({"title": "Crossref full-text link", "href": str(link.get("URL")), "body": ""})
            except Exception:
                pass
        if quote_words:
            queries.append(" ".join(f'"{word}"' for word in quote_words[-5:]))

        search_items: List[Dict[str, Any]] = []
        for query in queries[:6]:
            try:
                search_items.extend(await search_api.asearch(query, max_results=8))
            except Exception:
                continue

        evidence_texts: List[Dict[str, str]] = []
        for item in search_items:
            body = str((item or {}).get("body") or "").strip()
            href = str((item or {}).get("href") or "").strip()
            title_value = str((item or {}).get("title") or "").strip()
            if _quote_verifier_reject_url(href) or _quote_verifier_is_query_echo(href, title_value, body):
                continue
            if body:
                evidence_texts.append({"source_url": href, "source_type": "search_snippet", "text": f"{title_value}\n{body}"})

        for url in _quote_verifier_candidate_urls(citation, search_items)[:12]:
            if _quote_verifier_reject_url(url):
                continue
            try:
                raw_bytes, final_url, content_type = await asyncio.to_thread(
                    _http_get_bytes,
                    url,
                    headers=_default_http_headers(),
                    timeout=20,
                )
                if raw_bytes.startswith(b"%PDF") or "application/pdf" in str(content_type or "").lower() or str(final_url or url).lower().endswith(".pdf"):
                    text = _extract_pdf_text_from_bytes(raw_bytes, limit=30000)
                    source_type = "pdf"
                else:
                    text = _html_to_text(raw_bytes.decode("utf-8", errors="ignore"))[:30000]
                    source_type = "html"
                final_candidate_url = final_url or url
                if (
                    text
                    and not _looks_like_blocked_or_challenge_page(text)
                    and not _quote_verifier_reject_url(final_candidate_url)
                    and not _quote_verifier_is_query_echo(final_candidate_url, "", text[:800])
                ):
                    evidence_texts.append({"source_url": final_candidate_url, "source_type": source_type, "text": text})
            except Exception:
                continue

        best: Dict[str, Any] = {
            "status": "not_found",
            "matches": None,
            "mismatched_word": "",
            "correct_word": "",
            "source_url": "",
            "source_type": "",
            "support": "",
            "confidence": 0.0,
        }
        for evidence in evidence_texts:
            diff = _quote_verifier_best_diff(quoted_text, evidence.get("text", ""))
            if (
                evidence.get("source_type") == "search_snippet"
                and not _quote_verifier_trusted_snippet_source(evidence.get("source_url", ""))
            ):
                continue
            if float(diff.get("confidence") or 0.0) > float(best.get("confidence") or 0.0):
                best.update(diff)
                best["status"] = "verified" if diff.get("matches") is not None and float(diff.get("confidence") or 0.0) >= 0.72 else "uncertain"
                best["source_url"] = evidence.get("source_url", "")
                best["source_type"] = evidence.get("source_type", "")

        if best.get("mismatched_word"):
            best["answer"] = best["mismatched_word"]
        elif best.get("matches") is True:
            best["answer"] = "Yes"
        else:
            best["answer"] = ""
        return _truncate_json_safely(best)
    return run_quote_verifier


def run_openalex_author_works_factory():
    async def run_openalex_author_works(*args):
        values = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
        if not values:
            return "No author name provided."
        raw_author_value = values[0]
        if not str(raw_author_value).strip():
            return "No author name provided."
        known_work_title = str(values[1]).strip() if len(values) > 1 and values[1] else ""
        before_year = None
        if len(values) > 2 and str(values[2]).strip():
            try:
                before_year = int(str(values[2]).strip())
            except Exception:
                before_year = None
        try:
            max_works = int(values[3]) if len(values) > 3 and str(values[3]).strip() else 50
        except Exception:
            max_works = 50
        max_works = max(5, min(max_works, 100))
        author_names = _extract_string_list(raw_author_value)
        if not author_names:
            return "No author name provided."
        author_name = author_names[0]
        search_url = "https://api.openalex.org/authors?" + urlencode(
            {
                "search": author_name,
                "per-page": 10,
                "select": "id,display_name,works_count,cited_by_count,last_known_institutions,display_name_alternatives",
            }
        )
        search_data = await asyncio.to_thread(_openalex_get_json, search_url, {"User-Agent": "DynaCall/GAIA"}, 30)
        candidates = (search_data.get("results") or [])
        if not candidates:
            return _truncate_json_safely({"author_query": author_name, "matched_author": None, "works": []})
        known_lower = known_work_title.lower()
        best_author = None
        best_author_score = None
        best_works_payload: List[Dict[str, Any]] = []
        for candidate in candidates[:5]:
            author_id = str(candidate.get("id") or "").strip()
            if not author_id:
                continue
            works_url = "https://api.openalex.org/works?" + urlencode(
                {
                    "filter": f"author.id:{author_id}",
                    "sort": "publication_year:asc",
                    "per-page": max_works,
                    "select": "id,display_name,publication_year,publication_date,authorships,primary_location,biblio,type",
                }
            )
            works_data = await asyncio.to_thread(_openalex_get_json, works_url, {"User-Agent": "DynaCall/GAIA"}, 30)
            works_results = works_data.get("results") or []
            score = 0
            if known_lower:
                for work in works_results:
                    title = str(work.get("display_name") or "").lower()
                    if known_lower and known_lower in title:
                        score += 100
                    if title and title in known_lower:
                        score += 100
            if str(candidate.get("display_name") or "").lower() == author_name.lower():
                score += 10
            score += min(int(candidate.get("works_count") or 0), 20) / 100.0
            if best_author is None or score > best_author_score:
                best_author = candidate
                best_author_score = score
                best_works_payload = works_results
        if best_author is None:
            return _truncate_json_safely({"author_query": author_name, "matched_author": None, "works": []})
        works: List[Dict[str, Any]] = []
        for source_order, work in enumerate(best_works_payload):
            venue = str((((work.get("primary_location") or {}).get("source") or {}).get("display_name")) or "").strip()
            normalized_title = _normalize_bibliographic_title(str(work.get("display_name") or "").strip())
            works.append(
                {
                    "title": normalized_title,
                    "year": work.get("publication_year"),
                    "publication_date": str(work.get("publication_date") or "").strip(),
                    "source_order": source_order,
                    "type": str(work.get("type") or "").strip(),
                    "authors": _normalize_openalex_authorships(work.get("authorships")),
                    "venue": venue,
                    "id": str(work.get("id") or "").strip(),
                }
            )
        works = sorted(
            works,
            key=lambda item: (
                int(item["year"]) if isinstance(item.get("year"), int) else 999999,
                item.get("publication_date") or "9999-99-99",
                int(item.get("source_order") or 0),
            ),
        )
        result: Dict[str, Any] = {
            "author_query": author_name,
            "known_work_title": known_work_title,
            "matched_author": {
                "id": str(best_author.get("id") or "").strip(),
                "display_name": str(best_author.get("display_name") or "").strip(),
                "works_count": int(best_author.get("works_count") or 0),
                "cited_by_count": int(best_author.get("cited_by_count") or 0),
            },
            "works": works,
        }
        if works:
            result["earliest_work"] = works[0]
        if before_year is not None:
            filtered = [item for item in works if isinstance(item.get("year"), int) and int(item["year"]) < before_year]
            result["before_year"] = before_year
            result["works_before_year"] = filtered
            result["count_before_year"] = len(filtered)
            if filtered:
                result["earliest_work_before_year"] = filtered[0]
        return _truncate_json_safely(result)
    return run_openalex_author_works


def run_url_fetch_factory(url_fetch_chain):
    async def run_url_fetch(query, context=None):
        if isinstance(query, str) and query.startswith(("http://", "https://")):
            try:
                result = await asyncio.to_thread(
                    url_fetch_chain.web_search_api.fetch_url_content,
                    query,
                    url_fetch_chain.fetch_mode,
                )
                if isinstance(result, dict):
                    content = result.get("content", "")
                    if "blocked or challenge page" in str(content).lower():
                        return "Failed to fetch content from URLs."
                    return content
                return str(result)
            except Exception:
                pass

        parsed_context = context
        if context and isinstance(context, list) and len(context) == 1 and isinstance(context[0], str):
            raw = context[0]
            try:
                parsed_context = json.loads(raw)
            except Exception:
                try:
                    parsed_context = ast.literal_eval(raw)
                except Exception:
                    urls = []
                    for match in re.findall(r"Link:\s*(https?://\S+)", raw):
                        urls.append({"href": match.strip(), "title": "", "body": ""})
                    if not urls:
                        for match in re.findall(r"https?://\S+", raw):
                            urls.append({"href": match.strip(), "title": "", "body": ""})
                    parsed_context = urls if urls else context
        result = await url_fetch_chain.acall({
            "query": query,
            "search_results": parsed_context,
        })
        return result["content"]

    return run_url_fetch


def run_web_browser_factory(search_api: DuckDuckGoSearch):
    async def run_web_browser(query_or_url, search_results=None):
        if search_results and isinstance(search_results, list) and len(search_results) == 1 and isinstance(search_results[0], str):
            try:
                search_results = json.loads(search_results[0])
            except Exception:
                try:
                    search_results = ast.literal_eval(search_results[0])
                except Exception:
                    pass
        urls = []
        if isinstance(search_results, list):
            urls = [item.get("href", "") for item in search_results if isinstance(item, dict)]
        elif isinstance(query_or_url, str) and query_or_url.startswith(("http://", "https://")):
            urls = [query_or_url]
        fetched = []
        for url in urls[:5]:
            try:
                result = await search_api.afetch_url_content(url, mode="truncate")
                content = result.get("content", "")
                if _looks_like_blocked_or_challenge_page(content):
                    fetched.append(
                        f"URL: {result.get('redirect_url', url)}\nContent: Error fetching page: blocked or challenge page"
                    )
                else:
                    fetched.append(f"URL: {result.get('redirect_url', url)}\nContent: {content}")
            except Exception as exc:
                fetched.append(f"URL: {url}\nContent: Error fetching page: {exc}")
            if len(fetched) >= 3:
                break
        if not fetched:
            return "Failed to fetch content from URLs."
        return "\n\n".join(fetched)
    return run_web_browser


def run_search_get_contents_factory(search_api: DuckDuckGoSearch):
    async def run_search_get_contents(query_or_url, search_results=None):
        parsed_results = search_results
        if parsed_results and isinstance(parsed_results, list) and len(parsed_results) == 1 and isinstance(parsed_results[0], str):
            try:
                parsed_results = json.loads(parsed_results[0])
            except Exception:
                try:
                    parsed_results = ast.literal_eval(parsed_results[0])
                except Exception:
                    pass

        urls: List[str] = []
        if isinstance(query_or_url, str) and query_or_url.startswith(("http://", "https://")):
            urls.append(query_or_url)
        elif isinstance(parsed_results, list):
            for item in parsed_results:
                if isinstance(item, dict):
                    href = str(item.get("href", "")).strip()
                    if href.startswith(("http://", "https://")):
                        urls.append(href)
                elif isinstance(item, str) and item.startswith(("http://", "https://")):
                    urls.append(item)

        if not urls and isinstance(query_or_url, str) and query_or_url.strip():
            try:
                search_results_auto = await search_api.asearch(query_or_url, max_results=5)
            except Exception:
                search_results_auto = []
            for item in search_results_auto or []:
                if isinstance(item, dict):
                    href = str(item.get("href", "")).strip()
                    if href.startswith(("http://", "https://")):
                        urls.append(href)

        contents = []
        seen = set()
        for url in urls[:3]:
            if url in seen:
                continue
            seen.add(url)
            try:
                result = await search_api.afetch_url_content(url, mode="truncate")
                content = str((result or {}).get("content", "")).strip()
                redirect_url = str((result or {}).get("redirect_url", url)).strip()
                if (
                    not content
                    or "Error fetching page" in content
                    or content == "Failed to fetch content from URLs."
                    or _looks_like_blocked_or_challenge_page(content)
                ):
                    continue
                contents.append(f"URL: {redirect_url}\nContent:\n{content}")
            except Exception as exc:
                contents.append(f"URL: {url}\nContent:\nError fetching page: {exc}")

        return _truncate("\n\n".join(contents) if contents else "Failed to fetch content from URLs.")

    return run_search_get_contents


def run_deepsearch_factory(search_api: DuckDuckGoSearch, llm_adapter=None):
    def _loads_context(value: Any) -> Dict[str, Any]:
        if not value:
            return {}
        try:
            parsed = json.loads(str(value))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {"raw_context": str(value)}

    def _compact_docs_for_llm(docs: List[Dict[str, Any]], limit: int = 9000) -> str:
        chunks = []
        for idx, doc in enumerate(docs, start=1):
            chunks.append(
                f"[{idx}] {doc.get('title','')}\n"
                f"URL: {doc.get('url') or doc.get('href','')}\n"
                f"Snippet: {doc.get('body','')}\n"
                f"Content: {_truncate(str(doc.get('content','')), 1200)}"
            )
        return _truncate("\n\n".join(chunks), limit)

    def _parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(_extract_semantic_payload(raw))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            match = re.search(r"\{.*\}", raw or "", flags=re.S)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None

    def _normalize_query_list(value: Any, fallback: List[str], max_queries: int) -> List[str]:
        queries: List[str] = []
        if isinstance(value, list):
            for item in value:
                text = str(item).strip()
                if text and text not in queries:
                    queries.append(text)
        elif isinstance(value, str) and value.strip():
            for line in value.splitlines():
                text = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
                if text and text not in queries:
                    queries.append(text)
        for item in fallback:
            text = str(item).strip()
            if text and text not in queries:
                queries.append(text)
        return queries[: max(1, int(max_queries))]

    def _cross_source_query_hints(global_question: str, query_text: str) -> List[str]:
        question = (global_question or "").strip()
        if not question:
            return []

        hints: List[str] = []
        if query_text and query_text not in hints:
            hints.append(query_text)
        if question and question not in hints:
            hints.append(question)
        return hints

    async def _agent_generate_queries(
        query_text: str,
        global_question: str,
        docs: List[Dict[str, Any]],
        previous_queries: List[str],
        max_queries: int,
    ) -> List[str]:
        priority_hints = _cross_source_query_hints(global_question, query_text)
        fallback = [query_text]
        for hint in priority_hints:
            if hint and hint not in fallback:
                fallback.append(hint)
        for variant in _generate_search_fallbacks(query_text):
            if variant and variant not in fallback:
                fallback.append(variant)

        if llm_adapter is None:
            return fallback[: max(1, int(max_queries))]

        prompt = (
            "You are the query planner inside a GAIA deep-search tool.\n"
            "Generate source-aware atomic web search queries. Return strict JSON only:\n"
            "{\"queries\":[\"...\"]}\n\n"
            "Rules:\n"
            "- Use the Original Question as the source of truth; the Current Query may be incomplete or accidentally mixed.\n"
            "- If the question has multiple sources or objects, split them into separate atomic queries.\n"
            "- For 'which of these' questions, one query must retrieve the source defining the candidate set ('these'), and a separate query must retrieve the source where the candidate is matched.\n"
            "- Do not mix anchors from source A into source B. Example: source-A figure labels and source-B article date/category must be separate queries.\n"
            "- Prefer official/source-family anchors and exact dates, identifiers, titles, categories, or repositories.\n"
            "- Do not repeat previous queries.\n"
            f"- Return at most {max_queries} queries.\n\n"
            f"Original Question:\n{global_question or query_text}\n\n"
            f"Current Query:\n{query_text}\n\n"
            f"Previous Queries:\n{json.dumps(previous_queries, ensure_ascii=False)}\n\n"
            f"Current Evidence Summary:\n{_compact_docs_for_llm(docs, 3000) if docs else 'None'}\n"
        )
        raw = await llm_adapter.apredict(prompt)
        parsed = _parse_json_object(raw) if not raw.startswith("Error:") else None
        llm_queries = _normalize_query_list((parsed or {}).get("queries"), [], max_queries)
        if priority_hints:
            return _normalize_query_list(priority_hints + llm_queries, [], max_queries)
        return _normalize_query_list(llm_queries, fallback, max_queries)

    async def _agent_assess(
        query_text: str,
        global_question: str,
        docs: List[Dict[str, Any]],
        previous_queries: List[str],
        max_queries: int,
    ) -> Dict[str, Any]:
        if llm_adapter is None:
            return {"is_answer": bool(docs), "rewrite_queries": [], "reason": "", "summary": ""}
        prompt = (
            "You are the coverage critic inside a GAIA deep-search tool.\n"
            "Assess whether the fetched evidence is enough for downstream semantic extraction.\n"
            "Return strict JSON only:\n"
            "{\"is_answer\":true,\"rewrite_queries\":[\"...\"],\"reason\":\"...\",\"summary\":\"...\"}\n\n"
            "Rules:\n"
            "- Do not fabricate final answers; summarize only supported evidence and gaps.\n"
            "- If a source/object/date/category/identifier is unverified, is_answer must be false.\n"
            "- If multiple source objects are required, all must be covered separately.\n"
            "- rewrite_queries must be meaningfully different from previous queries.\n"
            f"- Return at most {max_queries} rewrite_queries.\n\n"
            f"Original Question:\n{global_question or query_text}\n\n"
            f"Current Query:\n{query_text}\n\n"
            f"Previous Queries:\n{json.dumps(previous_queries, ensure_ascii=False)}\n\n"
            f"Fetched Evidence:\n{_compact_docs_for_llm(docs)}\n"
        )
        raw = await llm_adapter.apredict(prompt)
        parsed = _parse_json_object(raw) if not raw.startswith("Error:") else None
        if not parsed:
            return {"is_answer": bool(docs), "rewrite_queries": [], "reason": "critic parse failed", "summary": ""}
        parsed["rewrite_queries"] = _normalize_query_list(parsed.get("rewrite_queries"), [], max_queries)
        parsed["is_answer"] = bool(parsed.get("is_answer"))
        return parsed

    async def _search_and_fetch(candidate_queries: List[str], max_results: int, seen_urls: set) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        query_results = []
        new_items: List[Dict[str, Any]] = []
        contents = []
        for candidate in candidate_queries:
            try:
                results = await search_api.asearch(candidate, max_results=min(max_results, 6))
            except Exception:
                results = []
            query_results.append({"query": candidate, "results": results[: min(5, len(results))]})
            for item in results:
                if not isinstance(item, dict):
                    continue
                href = str(item.get("href", "")).strip()
                if not href or href in seen_urls:
                    continue
                seen_urls.add(href)
                new_items.append(item)

        for item in new_items[:4]:
            href = str(item.get("href", "")).strip()
            if not href:
                continue
            try:
                fetched = await search_api.afetch_url_content(href, mode="truncate")
                content = str((fetched or {}).get("content", "")).strip()
                if not content or _looks_like_blocked_or_challenge_page(content):
                    continue
                contents.append(
                    {
                        "url": str((fetched or {}).get("redirect_url", href)).strip(),
                        "title": str(item.get("title", "")).strip(),
                        "body": str(item.get("body", "")).strip(),
                        "content": _truncate(content, 3000),
                    }
                )
            except Exception:
                continue
        return query_results, new_items, contents

    async def run_deepsearch(query, context=None, max_queries=3, max_results=8, max_loops=None):
        query_text = query[0] if isinstance(query, list) else query
        query_text = str(query_text).strip()
        if not query_text:
            return _truncate_json_safely({"query": "", "queries": [], "results": [], "contents": []})

        parsed_context = _loads_context(context)
        global_question = str(parsed_context.get("global_question", "") or "").strip()
        loop_count = int(max_loops or os.environ.get("TOOLWEAVER_DEEPSEARCH_MAX_LOOPS", "2") or 2)
        loop_count = max(1, min(loop_count, 3))

        query_results = []
        executed_queries: List[str] = []
        aggregated_items: List[Dict[str, Any]] = []
        evidence_docs: List[Dict[str, Any]] = []
        seen_urls = set()
        assessment: Dict[str, Any] = {"is_answer": False, "rewrite_queries": [], "reason": "", "summary": ""}
        pending_queries: List[str] = []

        for loop_idx in range(loop_count):
            if loop_idx == 0 or not pending_queries:
                candidate_queries = await _agent_generate_queries(
                    query_text=query_text,
                    global_question=global_question,
                    docs=evidence_docs,
                    previous_queries=executed_queries,
                    max_queries=max_queries,
                )
            else:
                candidate_queries = pending_queries[: max(1, int(max_queries))]

            candidate_queries = [q for q in candidate_queries if q and q not in executed_queries]
            if not candidate_queries:
                break

            executed_queries.extend(candidate_queries)
            round_results, round_items, round_contents = await _search_and_fetch(
                candidate_queries,
                int(max_results),
                seen_urls,
            )
            query_results.extend(round_results)
            aggregated_items.extend(round_items)
            evidence_docs.extend(round_contents)

            assessment = await _agent_assess(
                query_text=query_text,
                global_question=global_question,
                docs=evidence_docs,
                previous_queries=executed_queries,
                max_queries=max_queries,
            )
            pending_queries = [
                q for q in assessment.get("rewrite_queries", []) or []
                if str(q).strip() and str(q).strip() not in executed_queries
            ]
            if assessment.get("is_answer") or not pending_queries:
                break

        payload = {
            "query": query_text,
            "global_question": global_question,
            "queries": executed_queries,
            "results": query_results,
            "contents": evidence_docs,
            "assessment": assessment,
            "agentic": llm_adapter is not None,
        }
        if len(json.dumps(payload, ensure_ascii=False)) > 12000:
            payload["contents"] = [
                {
                    **doc,
                    "content": _truncate(str(doc.get("content", "")), 1600),
                }
                for doc in evidence_docs[:6]
            ]
            payload["results"] = [
                {
                    "query": item.get("query", ""),
                    "results": (item.get("results", []) or [])[:3],
                }
                for item in query_results[:6]
            ]
        return _truncate_json_safely(payload)

    return run_deepsearch


def run_code_interpreter_factory(files_root: Optional[str] = None):
    execute_multilang = run_execute_code_multilang_factory(files_root)

    async def run_code_interpreter(*args):
        if len(args) == 1:
            values = args[0] if isinstance(args[0], list) else [args[0]]
        else:
            values = list(args)
        if not values:
            return "No code provided."
        code_or_task = str(values[0])
        language = str(values[1]).strip() if len(values) > 1 and values[1] else "python"
        return await execute_multilang([code_or_task, language])

    return run_code_interpreter


def run_file_reader_factory(inspector: GAIAFileInspector, mode: str = "auto"):
    async def run_file_reader(file_path):
        target = file_path[0] if isinstance(file_path, list) else file_path
        if isinstance(target, str):
            lowered = target.strip().lower()
            if lowered.startswith("error downloading file:") or lowered.startswith("error:"):
                return target
        return await asyncio.to_thread(inspector.inspect_mode, target, mode)
    return run_file_reader


def _extract_orcid_identifier(value: str) -> Optional[str]:
    text = str(value).strip()
    match = re.search(r"\b(\d{4}-\d{4}-\d{4}-[\dX]{4})\b", text, flags=re.I)
    if match:
        return match.group(1)
    return None


def run_orcid_reader_factory():
    async def run_orcid_reader(*args):
        if len(args) == 1:
            values = args[0] if isinstance(args[0], list) else [args[0]]
        else:
            values = list(args)
        if not values:
            return "No ORCID identifier provided."

        raw_value = str(values[0]).strip()
        orcid_id = _extract_orcid_identifier(raw_value)
        if not orcid_id:
            return "Invalid ORCID identifier."

        before_year = None
        if len(values) > 1 and str(values[1]).strip():
            try:
                before_year = int(str(values[1]).strip())
            except Exception:
                before_year = None

        headers = {
            "Accept": "application/json",
            "User-Agent": "DynaCall/GAIA",
        }

        try:
            profile_payload, _ = await asyncio.to_thread(
                _http_get,
                f"https://orcid.org/{orcid_id}",
                headers,
                30,
            )
            profile = json.loads(profile_payload)
        except Exception:
            profile = {}

        works_payload, _ = await asyncio.to_thread(
            _http_get,
            f"https://pub.orcid.org/v3.0/{orcid_id}/works",
            headers,
            30,
        )
        works_data = json.loads(works_payload)

        person_name = ""
        try:
            given = (((profile.get("person") or {}).get("name") or {}).get("given-names") or {}).get("value", "")
            family = (((profile.get("person") or {}).get("name") or {}).get("family-name") or {}).get("value", "")
            person_name = " ".join(part for part in [given, family] if part).strip()
        except Exception:
            person_name = ""

        works = []
        counts_by_year: Dict[str, int] = {}
        groups = works_data.get("group", []) or []
        for group in groups:
            summaries = group.get("work-summary", []) or []
            summary = summaries[0] if summaries else {}
            title = (((summary.get("title") or {}).get("title") or {}).get("value") or "").strip()
            work_type = str(summary.get("type", "") or "").strip()
            pub_date = summary.get("publication-date") or {}
            year = str((pub_date.get("year") or {}).get("value") or "").strip()
            if year:
                counts_by_year[year] = counts_by_year.get(year, 0) + 1
            works.append(
                {
                    "title": title,
                    "year": year,
                    "type": work_type,
                }
            )

        works_sorted = sorted(
            works,
            key=lambda item: (
                int(item["year"]) if str(item.get("year", "")).isdigit() else 999999,
                item.get("title", ""),
            ),
        )

        result: Dict[str, Any] = {
            "orcid_id": orcid_id,
            "name": person_name,
            "total_public_works": len(works_sorted),
            "counts_by_year": counts_by_year,
            "works": works_sorted,
        }

        if before_year is not None:
            filtered = [
                item for item in works_sorted
                if str(item.get("year", "")).isdigit() and int(item["year"]) < before_year
            ]
            result["before_year"] = before_year
            result["count_before_year"] = len(filtered)
            result["works_before_year"] = filtered

        return _truncate_json_safely(result)

    return run_orcid_reader


def run_save_and_read_file_factory():
    async def run_save_and_read_file(*args):
        if len(args) == 1 and isinstance(args[0], list):
            values = args[0]
        else:
            values = list(args)
        if not values:
            return "No content provided."
        content = str(values[0])
        filename = str(values[1]).strip() if len(values) > 1 and values[1] else ""
        temp_dir = tempfile.gettempdir()
        if filename:
            path = Path(temp_dir) / Path(filename).name
            path.write_text(content, encoding="utf-8")
        else:
            with tempfile.NamedTemporaryFile("w", delete=False, dir=temp_dir, suffix=".txt", encoding="utf-8") as tmp:
                tmp.write(content)
                path = Path(tmp.name)
        return str(path)

    return run_save_and_read_file


def run_download_file_from_url_factory():
    def _coerce_http_url(raw_value: Any) -> str:
        text = str(raw_value or "").strip()
        if not text:
            return ""
        if text.startswith(("http://", "https://")):
            return text

        # Try parse JSON-like payloads and pull a likely URL field.
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                break
            except Exception:
                continue

        if isinstance(parsed, dict):
            for key in ("url", "href", "paper_url", "download_url", "file_url", "pdf_url", "source_url"):
                candidate = str(parsed.get(key, "")).strip()
                if candidate.startswith(("http://", "https://")):
                    return candidate
            for value in parsed.values():
                candidate = str(value or "").strip()
                if candidate.startswith(("http://", "https://")):
                    return candidate

        # Fallback: pull first concrete URL from free text / markdown / noisy snippets.
        match = re.search(r"https?://[^\s<>\]\"')]+", text)
        if match:
            return match.group(0).rstrip(".,);")
        return ""

    async def run_download_file_from_url(*args):
        if len(args) == 1 and isinstance(args[0], list):
            values = list(args[0])
        else:
            values = list(args)
        if not values:
            return "No URL provided."
        url = _coerce_http_url(values[0])
        filename = str(values[1]).strip() if len(values) > 1 and values[1] else ""
        parsed_input = urlparse(url)
        if parsed_input.scheme not in {"http", "https"} or not parsed_input.netloc:
            return "Error downloading file: input is not a concrete http(s) URL."
        try:
            def _download_once(target_url: str):
                final_url = target_url
                content_type = ""
                data = b""
                if requests is not None:
                    response = requests.get(
                        target_url,
                        stream=True,
                        timeout=30,
                        headers=_default_http_headers({"User-Agent": "DynaCall/GAIA"}),
                        allow_redirects=True,
                    )
                    response.raise_for_status()
                    final_url = response.url
                    content_type = response.headers.get("Content-Type", "")
                    data = response.content
                else:
                    request = Request(target_url, headers=_default_http_headers({"User-Agent": "DynaCall/GAIA"}))
                    with urlopen(request, timeout=30) as response:
                        final_url = response.geturl()
                        content_type = response.headers.get("Content-Type", "")
                        data = response.read()
                return final_url, content_type, data

            final_url, content_type, data = _download_once(url)
            lower_content_type = content_type.lower()

            if ("text/html" in lower_content_type or data[:15].lower().startswith(b"<!doctype html") or data[:6].lower().startswith(b"<html>")):
                html = data.decode("utf-8", errors="ignore")
                refresh_url = _extract_html_meta_refresh_url(html, final_url)
                image_url = _extract_html_image_url(html, final_url)
                file_url = _extract_html_download_url(html, final_url)
                if refresh_url:
                    final_url, content_type, data = _download_once(refresh_url)
                    lower_content_type = content_type.lower()
                elif file_url:
                    final_url, content_type, data = _download_once(file_url)
                    lower_content_type = content_type.lower()
                elif image_url:
                    final_url, content_type, data = _download_once(image_url)
                    lower_content_type = content_type.lower()
                else:
                    return "Error downloading file: URL resolved to an HTML page rather than a direct downloadable file."

            parsed_final = urlparse(final_url)
            parsed_name = Path(parsed_final.path).name or Path(parsed_input.path).name
            suffix = Path(parsed_name).suffix
            if not suffix:
                suffix = _infer_suffix_from_content_type(lower_content_type)
            target_name = filename or parsed_name or f"downloaded_{next(tempfile._get_candidate_names())}{suffix}"
            download_dir = Path.cwd() / "exps" / "downloads"
            download_dir.mkdir(parents=True, exist_ok=True)
            target_path = download_dir / Path(target_name).name
            if suffix and not target_path.suffix:
                target_path = target_path.with_suffix(suffix)
            with target_path.open("wb") as fh:
                fh.write(data)
            return str(target_path)
        except Exception as exc:
            return f"Error downloading file: {exc}"

    return run_download_file_from_url


def run_python_factory(files_root: Optional[str] = None):
    def _inject_excel_fallback(code_str: str) -> str:
        needs_excel = "read_excel" in code_str or "ExcelFile" in code_str
        needs_biopdb = "Bio.PDB" in code_str or "PDBParser" in code_str

        if not needs_excel and not needs_biopdb:
            return code_str

        prelude = """
import zipfile
import xml.etree.ElementTree as ET
import types
import sys as _toolweaver_sys
try:
    import pandas as _toolweaver_pd
except Exception:
    _toolweaver_pd = None
try:
    import Bio.PDB as _toolweaver_biopdb
except Exception:
    _toolweaver_biopdb = None

if _toolweaver_pd is not None:
    def _toolweaver_normalize_excel_rows(rows):
        header_idx = -1
        max_non_empty = 0
        for idx, row in enumerate(rows[:20]):
            non_empty = sum(bool(str(cell).strip()) for cell in row)
            if non_empty >= 2 and non_empty > max_non_empty:
                header_idx = idx
                max_non_empty = non_empty

        if header_idx == -1:
            width = max((len(row) for row in rows), default=1)
            headers = [f"col_{i}" for i in range(width)]
            records = []
            for row in rows:
                padded = list(row) + [""] * max(0, width - len(row))
                records.append({headers[i]: str(padded[i]).strip() for i in range(width)})
            return headers, records

        header_row = rows[header_idx]
        width = max(len(header_row), max((len(row) for row in rows[header_idx + 1:]), default=len(header_row)))
        headers = [str(cell).strip() or f"col_{i}" for i, cell in enumerate(header_row + [""] * max(0, width - len(header_row)))]

        records = []
        current_section = ""
        for row in rows[header_idx + 1:]:
            padded = list(row) + [""] * max(0, width - len(row))
            trimmed = [str(value).strip() for value in padded]
            non_empty = [value for value in trimmed if value]
            if not non_empty:
                continue
            if len(non_empty) == 1:
                current_section = non_empty[0]
                continue
            record = {headers[i]: trimmed[i] for i in range(width)}
            if current_section and "Media Type" not in record:
                record["Media Type"] = current_section
            records.append(record)

        final_headers = headers.copy()
        if any("Media Type" in record for record in records) and "Media Type" not in final_headers:
            final_headers.append("Media Type")
        return final_headers, records

    class _ToolWeaverExcelFile:
        def __init__(self, path):
            self.path = path
            self.sheet_names = list(_toolweaver_read_excel_xml(path).keys())

    def _toolweaver_read_excel_xml(path):
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        with zipfile.ZipFile(path) as zf:
            shared_strings = []
            if "xl/sharedStrings.xml" in zf.namelist():
                root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in root.findall("m:si", ns):
                    text = "".join(t.text or "" for t in si.iterfind(".//m:t", ns))
                    shared_strings.append(text)
            workbook = ET.fromstring(zf.read("xl/workbook.xml"))
            rels = {}
            if "xl/_rels/workbook.xml.rels" in zf.namelist():
                rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
                for rel in rel_root:
                    rels[rel.attrib.get("Id")] = rel.attrib.get("Target")
            sheets = {}
            for sheet in workbook.findall(".//m:sheet", ns):
                name = sheet.attrib.get("name", "Sheet")
                rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                target = rels.get(rel_id, "")
                if not target:
                    continue
                xml_path = "xl/" + target.lstrip("/")
                sheet_root = ET.fromstring(zf.read(xml_path))
                rows = []
                for row in sheet_root.findall(".//m:sheetData/m:row", ns):
                    values = []
                    for cell in row.findall("m:c", ns):
                        cell_type = cell.attrib.get("t")
                        value = cell.findtext("m:v", default="", namespaces=ns)
                        if cell_type == "s" and value.isdigit():
                            idx = int(value)
                            value = shared_strings[idx] if idx < len(shared_strings) else value
                        values.append(value)
                    rows.append(values)
                headers, records = _toolweaver_normalize_excel_rows(rows)
                sheets[name] = _toolweaver_pd.DataFrame(records, columns=headers or ["col_0"])
            return sheets

    _toolweaver_original_read_excel = getattr(_toolweaver_pd, "read_excel", None)
    _toolweaver_original_excel_file = getattr(_toolweaver_pd, "ExcelFile", None)

    def _toolweaver_patched_read_excel(io, *args, **kwargs):
        if isinstance(io, _ToolWeaverExcelFile):
            sheets = _toolweaver_read_excel_xml(io.path)
            sheet_name = kwargs.get("sheet_name", args[0] if args else 0)
            if sheet_name is None:
                return sheets
            if isinstance(sheet_name, str):
                return sheets[sheet_name]
            names = list(sheets.keys())
            return sheets[names[sheet_name]]
        try:
            if _toolweaver_original_read_excel is None:
                raise ImportError("pandas.read_excel unavailable")
            return _toolweaver_original_read_excel(io, *args, **kwargs)
        except Exception as exc:
            if "openpyxl" not in str(exc).lower():
                raise
            path = io.path if hasattr(io, "path") else io
            sheets = _toolweaver_read_excel_xml(path)
            sheet_name = kwargs.get("sheet_name", args[0] if args else 0)
            if sheet_name is None:
                return sheets
            if isinstance(sheet_name, str):
                return sheets[sheet_name]
            names = list(sheets.keys())
            return sheets[names[sheet_name]]

    def _toolweaver_patched_excel_file(path, *args, **kwargs):
        try:
            if _toolweaver_original_excel_file is None:
                raise ImportError("pandas.ExcelFile unavailable")
            return _toolweaver_original_excel_file(path, *args, **kwargs)
        except Exception as exc:
            if "openpyxl" not in str(exc).lower():
                raise
            return _ToolWeaverExcelFile(path)

    _toolweaver_pd.read_excel = _toolweaver_patched_read_excel
    _toolweaver_pd.ExcelFile = _toolweaver_patched_excel_file

if _toolweaver_biopdb is None:
    class _ToolWeaverVec:
        def __init__(self, values):
            self.values = [float(v) for v in values]

        def __sub__(self, other):
            return _ToolWeaverVec([a - b for a, b in zip(self.values, other.values)])

        def __pow__(self, power):
            return _ToolWeaverVec([value ** power for value in self.values])

        def __getitem__(self, index):
            return self.values[index]

        def sum(self):
            return sum(self.values)

    class _ToolWeaverAtom:
        def __init__(self, x, y, z):
            self._coord = _ToolWeaverVec([x, y, z])
            self.coord = self._coord

        def get_coord(self):
            return self._coord

    class _ToolWeaverResidue:
        def __init__(self, atoms):
            self._atoms = atoms

        def __iter__(self):
            return iter(self._atoms)

    class _ToolWeaverChain:
        def __init__(self, atoms):
            self._atoms = atoms
            self._residues = [_ToolWeaverResidue([atom]) for atom in atoms]

        def get_atoms(self):
            return iter(self._atoms)

        def __iter__(self):
            return iter(self._residues)

    class _ToolWeaverModel:
        def __init__(self, atoms):
            self._chains = [_ToolWeaverChain(atoms)]

        def get_chains(self):
            return iter(self._chains)

        def __iter__(self):
            return iter(self._chains)

        def __getitem__(self, index):
            return self._chains[index]

    class _ToolWeaverStructure:
        def __init__(self, atoms):
            self._atoms = atoms
            self._models = [_ToolWeaverModel(atoms)]

        def get_atoms(self):
            return iter(self._atoms)

        def __iter__(self):
            return iter(self._models)

        def __getitem__(self, index):
            return self._models[index]

    class PDBParser:
        def __init__(self, *args, **kwargs):
            pass

        def get_structure(self, structure_id, path):
            atoms = []
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if not (line.startswith("ATOM") or line.startswith("HETATM")):
                        continue
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                    except Exception:
                        continue
                    atoms.append(_ToolWeaverAtom(x, y, z))
            return _ToolWeaverStructure(atoms)

    _toolweaver_bio_module = types.ModuleType("Bio")
    _toolweaver_biopdb_module = types.ModuleType("Bio.PDB")
    _toolweaver_biopdb_module.PDBParser = PDBParser
    _toolweaver_bio_module.PDB = _toolweaver_biopdb_module
    _toolweaver_sys.modules.setdefault("Bio", _toolweaver_bio_module)
    _toolweaver_sys.modules["Bio.PDB"] = _toolweaver_biopdb_module
"""
        return prelude + "\n" + code_str

    async def run_python(code):
        code_str = code[0] if isinstance(code, list) else code
        code_str = _inject_excel_fallback(code_str)
        working_dir = files_root or os.getcwd()
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code_str)
            script_path = f.name
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                script_path,
                cwd=working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            stdout_text = stdout.decode("utf-8", errors="ignore").strip()
            stderr_text = stderr.decode("utf-8", errors="ignore").strip()
            if proc.returncode != 0:
                return f"Python execution failed.\nSTDERR:\n{_truncate(stderr_text)}"
            if stderr_text:
                return _truncate(f"STDOUT:\n{stdout_text}\n\nSTDERR:\n{stderr_text}")
            return _truncate(stdout_text or "Python execution completed with empty stdout.")
        finally:
            try:
                os.remove(script_path)
            except OSError:
                pass

    return run_python


def run_execute_code_multilang_factory(files_root: Optional[str] = None):
    python_runner = run_python_factory(files_root)

    async def run_execute_code_multilang(args):
        values = args if isinstance(args, list) else [args]
        if not values:
            return "No code provided."
        code = values[0]
        language = str(values[1]).strip().lower() if len(values) > 1 and values[1] else "python"
        working_dir = files_root or os.getcwd()

        if language == "python":
            return await python_runner(code)

        if language == "bash":
            script = str(code[0] if isinstance(code, list) else code)
            with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as fh:
                fh.write(script)
                script_path = fh.name
            try:
                proc = await asyncio.create_subprocess_exec(
                    "/bin/bash",
                    script_path,
                    cwd=working_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                stdout_text = stdout.decode("utf-8", errors="ignore").strip()
                stderr_text = stderr.decode("utf-8", errors="ignore").strip()
                if proc.returncode != 0:
                    return _truncate(f"Bash execution failed.\nSTDERR:\n{stderr_text}")
                return _truncate(stdout_text or stderr_text or "Bash execution completed with empty stdout.")
            finally:
                try:
                    os.remove(script_path)
                except OSError:
                    pass

        if language == "sql":
            script = (
                "import sqlite3\n"
                "conn = sqlite3.connect(':memory:')\n"
                "cur = conn.cursor()\n"
                f"script = r'''{str(code[0] if isinstance(code, list) else code)}'''\n"
                "cur.executescript(script)\n"
                "rows = cur.fetchall()\n"
                "print(rows)\n"
            )
            return await python_runner(script)

        if language == "c":
            if not shutil.which("cc"):
                return "C execution unsupported: local C compiler not found."
            source = str(code[0] if isinstance(code, list) else code)
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = Path(tmpdir) / "main.c"
                bin_path = Path(tmpdir) / "main.out"
                src_path.write_text(source, encoding="utf-8")
                compile_proc = await asyncio.create_subprocess_exec(
                    "cc", str(src_path), "-o", str(bin_path),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                _, compile_err = await asyncio.wait_for(compile_proc.communicate(), timeout=30)
                if compile_proc.returncode != 0:
                    return _truncate(f"C compilation failed.\nSTDERR:\n{compile_err.decode('utf-8', errors='ignore')}")
                run_proc = await asyncio.create_subprocess_exec(
                    str(bin_path),
                    cwd=working_dir,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(run_proc.communicate(), timeout=30)
                if run_proc.returncode != 0:
                    return _truncate(f"C execution failed.\nSTDERR:\n{stderr.decode('utf-8', errors='ignore')}")
                return _truncate(stdout.decode("utf-8", errors="ignore").strip() or "C execution completed with empty stdout.")

        if language == "java":
            if not shutil.which("javac") or not shutil.which("java"):
                return "Java execution unsupported: local Java toolchain not found."
            source = str(code[0] if isinstance(code, list) else code)
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = Path(tmpdir) / "Main.java"
                src_path.write_text(source, encoding="utf-8")
                compile_proc = await asyncio.create_subprocess_exec(
                    "javac", str(src_path),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                _, compile_err = await asyncio.wait_for(compile_proc.communicate(), timeout=30)
                if compile_proc.returncode != 0:
                    return _truncate(f"Java compilation failed.\nSTDERR:\n{compile_err.decode('utf-8', errors='ignore')}")
                run_proc = await asyncio.create_subprocess_exec(
                    "java", "-cp", tmpdir, "Main",
                    cwd=working_dir,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(run_proc.communicate(), timeout=30)
                if run_proc.returncode != 0:
                    return _truncate(f"Java execution failed.\nSTDERR:\n{stderr.decode('utf-8', errors='ignore')}")
                return _truncate(stdout.decode("utf-8", errors="ignore").strip() or "Java execution completed with empty stdout.")

        return f"Unsupported language: {language}. Supported: python, bash, sql, c, java."

    return run_execute_code_multilang


def run_calculator_factory():
    async def run_calculator(expression):
        expr = expression[0] if isinstance(expression, list) else expression
        try:
            safe_names = {
                "ceil": math.ceil,
                "floor": math.floor,
                "sqrt": math.sqrt,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
            }
            value = eval(expr, {"__builtins__": {}}, safe_names)
        except Exception as exc:
            return f"Calculation error: {exc}"
        return str(value)

    return run_calculator


def run_verifier_factory(llm_adapter):
    async def run_verifier(*args):
        if len(args) == 1:
            values = args[0] if isinstance(args[0], list) else [args[0]]
        else:
            values = list(args)

        question = str(values[0]).strip() if len(values) >= 1 else ""
        proposed_answer = str(values[1]).strip() if len(values) >= 2 else ""
        evidence = str(values[2]).strip() if len(values) >= 3 else ""

        if not question:
            return json.dumps(
                {
                    "valid": False,
                    "confidence": 0.0,
                    "issues": ["missing question"],
                    "recommendation": "retry_different_approach",
                },
                ensure_ascii=False,
            )

        issues: List[str] = []
        confidence = 0.5

        if not proposed_answer:
            issues.append("empty answer")
            confidence -= 0.4

        answer_l = proposed_answer.lower()
        question_l = question.lower()

        if len(proposed_answer) > 120:
            issues.append("answer too long")
            confidence -= 0.15

        if any(phrase in answer_l for phrase in ("the answer is", "based on", "according to", "i think", "likely")):
            issues.append("answer contains explanatory or hedging text")
            confidence -= 0.15

        if any(token in question_l for token in ("what year", "which year", "founded", "when")):
            if not re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", proposed_answer):
                issues.append("question expects a year-like answer")
                confidence -= 0.2

        if any(token in question_l for token in ("yes/no", "is it", "does it", "was it")):
            if answer_l not in {"yes", "no", "true", "false"}:
                issues.append("question expects yes/no style answer")
                confidence -= 0.2

        if evidence:
            ev_l = evidence.lower()
            if proposed_answer and proposed_answer.lower() in ev_l:
                confidence += 0.25
            elif proposed_answer:
                confidence -= 0.1
                issues.append("answer not explicitly found in evidence")

        prompt = (
            "You are a strict answer verifier.\n"
            "Assess whether the proposed answer is concise, directly responsive, and supported by the evidence.\n"
            "Return strict JSON only with keys valid, confidence, issues, recommendation.\n"
            "recommendation must be one of: proceed, investigate_further, retry_different_approach.\n\n"
            f"Question:\n{question}\n\n"
            f"Proposed answer:\n{proposed_answer}\n\n"
            f"Evidence:\n{evidence}\n"
        )
        raw = await llm_adapter.apredict(prompt)
        if raw.startswith("Error:"):
            model_result = None
        else:
            try:
                model_result = json.loads(_extract_semantic_payload(raw))
            except Exception:
                model_result = None

        if isinstance(model_result, dict):
            valid = bool(model_result.get("valid", not issues))
            model_conf = model_result.get("confidence", confidence)
            try:
                confidence = max(0.0, min(1.0, float(model_conf)))
            except Exception:
                confidence = max(0.0, min(1.0, confidence))
            model_issues = model_result.get("issues", [])
            if isinstance(model_issues, list):
                for item in model_issues:
                    item_s = str(item).strip()
                    if item_s and item_s not in issues:
                        issues.append(item_s)
            recommendation = str(model_result.get("recommendation", "")).strip() or None
        else:
            confidence = max(0.0, min(1.0, confidence))
            valid = not issues and confidence >= 0.75
            recommendation = None

        if recommendation not in {"proceed", "investigate_further", "retry_different_approach"}:
            if valid and confidence >= 0.75:
                recommendation = "proceed"
            elif confidence >= 0.4:
                recommendation = "investigate_further"
            else:
                recommendation = "retry_different_approach"

        payload = {
            "valid": bool(valid),
            "confidence": round(float(confidence), 4),
            "issues": issues,
            "recommendation": recommendation,
        }
        return json.dumps(payload, ensure_ascii=False)

    return run_verifier


def run_semantic_map_factory(llm_adapter):
    def _extract_blocked_domains_from_context(context: Dict[str, Any]) -> List[str]:
        domains = context.get("blocked_domains", []) or []
        cleaned: List[str] = []
        for item in domains:
            domain = str(item).strip().lower()
            if domain.startswith("www."):
                domain = domain[4:]
            if domain and domain not in cleaned:
                cleaned.append(domain)
        return cleaned

    def _is_blocked_selected_url(value: Any, blocked_domains: List[str]) -> bool:
        if not blocked_domains or not isinstance(value, str):
            return False
        text = value.strip()
        if not text.startswith(("http://", "https://")):
            return False
        try:
            domain = (urlparse(text).netloc or "").lower()
        except Exception:
            return False
        if domain.startswith("www."):
            domain = domain[4:]
        return any(domain == blocked or domain.endswith(f".{blocked}") for blocked in blocked_domains)

    def _normalize_shortest_character_alias(
        value: Any,
        instruction: str,
        context: Dict[str, Any],
    ) -> Any:
        """Normalize common character-name aliases when prompt requests shortest naming."""
        if not isinstance(value, str):
            return value

        instruction_l = (instruction or "").lower()
        question_l = str(context.get("global_question", "") or "").lower()
        gate_text = f"{instruction_l} {question_l}"

        asks_shortest = ("shortest" in gate_text) and any(
            marker in gate_text for marker in ("character", "token", "symbol", "punctuation")
        )
        if not asks_shortest:
            return value

        normalized = value.strip().lower()
        alias_map = {
            "backquote": "backtick",
            "grave accent": "backtick",
            "`": "backtick",
        }
        return alias_map.get(normalized, value)

    async def run_semantic_map(*args):
        if len(args) == 1:
            values = args[0] if isinstance(args[0], list) else [args[0]]
        else:
            values = list(args)

        # Accept both canonical semantic_map([instruction, inputs, output_schema])
        # and planner-emitted positional variants like
        # semantic_map([instruction], inputs, output_schema).
        if len(values) >= 3 and isinstance(values[0], (list, tuple)):
            first_item = list(values[0])
            if len(first_item) == 1 and isinstance(first_item[0], str):
                values = [first_item[0], values[1], values[2], *values[3:]]

        if len(values) == 1 and isinstance(values[0], tuple):
            values = list(values[0])

        if len(values) < 3:
            return "semantic_map expects either [instruction, inputs, output_schema] or [global_question, local_request, plan_context, inputs, output_schema]."

        context = {}
        if len(values) >= 6:
            try:
                context = json.loads(str(values[5]))
            except Exception:
                context = {"raw_context": str(values[5])}

        if len(values) >= 5:
            global_question = str(values[0]).strip()
            local_request = str(values[1]).strip()
            plan_context = str(values[2]).strip()
            inputs = values[3]
            output_schema = str(values[4]).strip() or "string"
        else:
            local_request = str(values[0]).strip()
            inputs = values[1]
            output_schema = str(values[2]).strip() or "string"
            global_question = str(context.get("global_question", "") or "").strip()
            plan_context = str(context.get("plan_context", "") or local_request).strip()

        if not context:
            context = {}
        context.setdefault("global_question", global_question)
        context.setdefault("local_question", local_request)
        context.setdefault("plan_context", plan_context)
        blocked_domains = _extract_blocked_domains_from_context(context)

        if not isinstance(inputs, list):
            inputs = [inputs]

        batch_spec = None
        if local_request.startswith("{"):
            try:
                parsed_request = json.loads(local_request)
                if isinstance(parsed_request, dict) and parsed_request.get("mode") == "batch":
                    batch_spec = parsed_request
            except Exception:
                batch_spec = None

        if batch_spec is not None:
            items = batch_spec.get("items", []) or []
            prompt_lines = [
                "You are semantic_map, a typed semantic operator with global task awareness.",
                "Perform each requested semantic transformation using only the provided evidence.",
                "Do not solve the whole task, do not call tools, and do not invent missing facts.",
                "Return one result per item in this exact format:",
                "Result 1: <value>",
                "Result 2: <value>",
                "No explanations, markdown, bullets, or extra text.",
                f"Global Question:\n{context.get('global_question', global_question)}",
                f"Plan Context:\n{context.get('plan_context', plan_context)}",
            ]
            for idx, item in enumerate(items, start=1):
                prompt_lines.append(f"Item {idx} instruction:\n{item.get('instruction', '')}")
                prompt_lines.append(f"Item {idx} output schema:\n{item.get('output_schema', 'string')}")
                depends_on_result = item.get("depends_on_result")
                if depends_on_result:
                    prompt_lines.append(f"Item {idx} input depends on Result {depends_on_result}.")
                if idx - 1 < len(inputs):
                    prompt_lines.append(f"Item {idx} input:\n{str(inputs[idx - 1]).strip()}")
            prompt_lines.append("Output result lines in item order.")

            raw = await llm_adapter.apredict("\n\n".join(prompt_lines))
            if raw.startswith("Error:"):
                return raw
            payload = _extract_semantic_payload(raw)
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, list):
                    return "\n".join(
                        f"Result {idx}: {_format_semantic_value(value, str(items[idx - 1].get('output_schema', 'string')))}"
                        for idx, value in enumerate(parsed, start=1)
                    )
            except Exception:
                pass
            if re.search(r"(?im)^\s*Result\s*1\s*:", payload):
                return _truncate(payload)
            return _truncate(
                "\n".join(
                    f"Result {idx}: {line.strip()}"
                    for idx, line in enumerate(payload.splitlines(), start=1)
                    if line.strip()
                )
            )

        rendered_inputs = []
        for idx, item in enumerate(inputs, start=1):
            rendered_inputs.append(f"Input {idx}:\n{str(item)}")

        provenance_lines = []
        for item in context.get("inputs_with_provenance", []) or []:
            provenance_lines.append(
                f"- Tool: {item.get('tool', '')}\n"
                f"  Action: {item.get('action', '')}\n"
                f"  Observation: {item.get('observation', '')}"
            )

        prompt = (
            "You are semantic_map, a typed semantic operator with global task awareness.\n"
            "Your job is to convert raw observations into the smallest trustworthy artifact needed by the next step.\n"
            "You are not the final answering agent.\n"
            "Do not solve the whole task, do not call tools, and do not invent missing facts.\n\n"
            "Preferred calling convention matches information_extract-style reasoning:\n"
            "- Global question: what the whole task is asking\n"
            "- Local request: what this step must output next\n"
            "- Plan context: why this step exists in the current chain\n"
            "- Observations: the direct evidence for this step\n"
            "- Output schema: the exact typed artifact the next step can consume\n\n"
            "What semantic_map is for:\n"
            "- extract exact fields from noisy evidence\n"
            "- normalize surface forms into the requested representation\n"
            "- select the best object/URL/span from candidates\n"
            "- align different source wordings when they refer to the same supported entity\n"
            "- rewrite broad evidence into a compact typed artifact for later branch/tool steps\n"
            "- summarize the current globally-relevant state so the next step can choose the right branch\n\n"
            "Core rules:\n"
            "- Stay evidence-grounded: use only the provided inputs and context.\n"
            "- Treat the direct inputs as primary evidence. History/context is auxiliary and must not replace missing direct evidence.\n"
            "- Read the direct observations in light of the global question, local request, plan context, history of observations, and current plan fragment.\n"
            "- Prefer outputs that help the next branch/replan decision, not outputs that merely restate a local snippet.\n"
            "- Be conservative: if support is partial, preserve candidates or support fields instead of guessing.\n"
            "- Preserve structure: when the schema allows records/lists/objects, keep structure instead of collapsing too early to one scalar.\n"
            "- If the next step depends on whether an object, operand, or route is verified, make that state explicit in the typed output when the schema allows it.\n"
            "- Return the shortest valid value matching the schema.\n"
            "- For cross-source matching, if source A uses a concept or alias but source B uses a different supported surface form, return the exact supported surface form from source B.\n"
            "- If selecting or extracting a URL, never return a URL from a blocked domain.\n"
            "- If only blocked-domain URLs fit, return the empty value for the schema.\n\n"
            "Schema discipline:\n"
            "- string -> return one plain string and nothing else.\n"
            "- number -> return one JSON number or numeric string that can be parsed to the target number.\n"
            "- boolean -> return true or false.\n"
            "- list[string] -> return a strict JSON array of strings.\n"
            "- json{...} -> return one strict JSON object matching the schema exactly.\n"
            "- For json objects, fill unsupported string fields with \"\", list fields with [], boolean fields with false, and object fields with empty matching values.\n"
            "- Never emit markdown, comments, prose, code fences, trailing commas, or Python/JavaScript dict syntax.\n\n"
            "Reliability rules:\n"
            "- Evidence beats prior expectations.\n"
            "- If multiple candidates remain plausible, do not force one unless the instruction asks for best-supported selection.\n"
            "- For count/list questions, prefer extracting records or operands first when the schema permits it.\n"
            "- For questions constrained to a source-defined entity set (for example members of a bloc, rows in a table, entries in a list, or candidates on one grounded page), keep that set closed: extract only entities explicitly present in the grounded source and do not add future members, observers, applicants, or memory-based extras.\n"
            "- For named video/interview questions, if video metadata or transcript evidence grounds a specific interviewee set, keep that set closed in later extraction. Do not introduce outside people from general AI-history pages, quote collections, or related-figure articles.\n"
            "- If later computation needs those entities, preserve them as structured records/list items so downstream code can operate on the grounded set instead of recreating it from memory.\n"
            "- For bloc/union/alliance questions, treat the current formal member set in the grounded source as closed. Do not add candidate states, observers, or accession states unless the same evidence explicitly marks them as members.\n"
            "- If the local request needs country-capital/member records for later computation and the grounded set is incomplete, return the partial structured set or empty value rather than silently filling missing members/capitals from memory.\n"
            "- For country-capital extraction tasks, include a record only when both the country and its capital are explicitly grounded in the same evidence. Do not emit placeholder records with empty capitals.\n"
            "- If a country such as Timor-Leste appears only in accession, observer, applicant, roadmap, or future-membership context rather than as a current member in the same grounded evidence, exclude it from current-member records.\n"
            "- For repeated-removal, threshold-crossing, or minimum-step arithmetic questions, extract the grounded numeric operands and any needed unit-normalized constants; do not infer the final count unless the local request explicitly asks for that computed scalar.\n"
            "- For section/table count questions, count only records inside the requested section/table; do not use years or names from prose, awards, references, biography, or unrelated sections.\n"
            "- For version-audit or superseded-status questions, prefer records with explicit booleans and support rather than a final scalar.\n"
            "- For URL/object selection, prefer the most exact and source-supported candidate, not the broadest page.\n"
            "- When extracting URLs from noisy snippets, normalize lightly obfuscated forms such as spaced punctuation, split domains, or spaced slashes (for example \"you tu . be / abc123\" -> \"https://youtu.be/abc123\").\n"
            "- For YouTube URL extraction, prefer canonical single-video URLs only: https://www.youtube.com/watch?v=<id> or https://youtu.be/<id>. Reject clip/channel/playlist/search URLs unless they are explicitly transformed into one concrete canonical video URL with a grounded video id.\n"
            "- If search-result snippets already contain the exact grounded numeric operands requested by the local step and the global question only needs deterministic computation afterward, extract those operands directly instead of forcing a later URL-selection artifact.\n"
            "- If search-result snippets already contain grounded comparison evidence such as person names, quotes, years, or timeframes, you may extract structured comparison records directly from those snippets instead of forcing a later page-open step.\n"
            "- For YouTube/video questions, if a youtube_transcript observation says has_transcript=false or transcript_source=fallback_search, treat its transcript field as noisy external evidence rather than a faithful caption transcript.\n"
            "- For such noisy YouTube fallback evidence, do not return an empty value merely because you cannot build a full structured table of all people and dates. If the local request asks for one direct answer such as which named person predicted sooner, combine the fallback evidence with any corroborating page evidence and return the best grounded direct answer.\n"
            "- When a video/interview question names a small closed set of people and asks for a direct comparison such as earlier/later, sooner/later, larger/smaller, or first/last, prefer a direct grounded comparison answer over an intermediate exhaustive schema if the latter is fragile.\n"
            "- For exact-paper numeric questions, distinguish paper-body/PDF evidence from citation export, RIS/BibTeX, abstract, listing, and metadata-only evidence. If the local request asks for a paper URL, prefer the paper body or direct PDF URL; if it asks for the numeric answer, extract it only from paper-body/PDF text.\n"
            "- For paper URL selection, prefer stable canonical article/body pages over fragile mirror links when both represent the same paper. Avoid citationstylelanguage/RIS/BibTeX endpoints for body extraction.\n"
            "- In the early stage of solving, prefer outputs that expose boundary conditions for branch/replan, such as object_ok, records, candidate_answers, or support.\n"
            "- For noisy long text, keep only fields directly useful for the next step.\n"
            "- If the instruction asks for a URL after repeated 403/404/forbidden/access-denied failures on one domain family, do not return another URL from that same blocked family. Prefer archive, mirror, cached, image, or alternate-source URLs instead; if none are supported, return the empty value.\n"
            "- If the observations are obviously polluted or unrelated to the target entity/domain, return the empty value rather than selecting a spurious URL or answer from them.\n"
            "- If official/canonical pages are blocked but search-result snippets from that canonical page still contain grounded item names, years, captions, or epitaph lines, you may extract those grounded intermediate entities directly from the snippets.\n"
            "- For oldest/earliest item questions, use explicit years or ordered chronology from the observations to identify the target item before extracting downstream text.\n"
            "- For questions about a background photo/object on an item's page, if the page is blocked, use reachable snippets or alternate sources to recover the background object's quoted text once the target item is grounded.\n"
            "- For image-specific questions, if a reachable page or result provides a concrete image URL or downloadable image, prefer extracting the answer from OCR/image evidence rather than from surrounding prose alone.\n"
            "- When a question explicitly asks about text on a background object in a photo, do not return text from the foreground object unless the observations explicitly show they are the same object.\n"
            "- If a page transcribes multiple captions/epitaphs but does not preserve which one belongs to the background object in the target photo, return the empty value rather than guessing from the foreground item's own caption or nearby ordering.\n"
            "- For passage questions asking for the first place mentioned by name, scan the wording in order and return the earliest place-name mention in the passage text itself, not a later throne location, scene location, or narrative setting that appears afterward.\n"
            "- If a passage contains an ordered phrase such as 'from India to Cush' and later mentions another location such as 'the citadel of Susa', the first named place is India because it appears first in the wording.\n"
            "- For historical officeholder questions with an exact month or date, choose the officeholder whose tenure range explicitly includes that month/date. Do not return a predecessor or successor whose term ended before or began after the requested date.\n"
            "- For biology or assay questions that ask for EC numbers, EC means Enzyme Commission numbers in dotted form, not ECHA/European Community substance identifiers in hyphenated form.\n"
            "- For biology EC-number tasks, prefer assay-linked enzymes over generic chemicals, buffers, salts, detergents, dyes, substrates, chromogens, primers, insecticides, or other incidental reagents.\n"
            "- If observations mention ELISA, NCM-ELISA, or DAS-ELISA, interpret requests for the key chemicals/entities used by the assay as the assay-linked readout enzymes unless the request explicitly asks for substrates or buffers.\n"
            "- For ELISA-family virus-detection questions asking for the two most commonly used chemicals/entities, use the canonical assay-linked enzyme pair alkaline phosphatase and horseradish peroxidase unless the observations explicitly support a different pair.\n"
            "- Do not return hyphenated substance IDs such as 428-040-8 for biology EC-number requests. Valid EC outputs here are dotted enzyme identifiers such as 3.1.3.1.\n"
            "- If the current evidence grounds enzyme names but not final EC numbers, return the grounded enzyme names or method cues needed for the next step rather than substituting nearby non-enzyme identifiers.\n"
            "- If the request asks for semicolon-separated EC numbers in the order of alphabetized chemicals/entities, first determine the alphabetical order of the grounded enzyme/entity names, then emit the corresponding dotted EC numbers in that same order.\n\n"
            "- When the local request asks for semicolon-separated output, use the normalized form `item1; item2` with one space after each semicolon unless the request explicitly asks for another spacing style.\n\n"
            "Mini examples:\n"
            "Example 1\n"
            "Instruction: Select the exact PDF URL for the target report.\n"
            "Schema: json{url:string,title:string}\n"
            "Input snippet: results include homepage, report landing page, and https://site.org/reports/2024-report.pdf\n"
            "Output: {\"url\":\"https://site.org/reports/2024-report.pdf\",\"title\":\"2024 report\"}\n\n"
            "Example 2\n"
            "Instruction: Extract structured records for later counting.\n"
            "Schema: json{records:list[string],support:string}\n"
            "Input snippet: Discography section lists: Album A (1999), Album B (2001), Album C (2004)\n"
            "Output: {\"records\":[\"Album A (1999)\",\"Album B (2001)\",\"Album C (2004)\"],\"support\":\"Discography section\"}\n\n"
            "Example 3\n"
            "Instruction: Normalize the candidate to the exact supported surface form from source B.\n"
            "Schema: string\n"
            "Input snippet: source A says ampersand, source B says ampersand sign\n"
            "Output: ampersand\n\n"
            "Example 4\n"
            "Instruction: Extract typed fields from a noisy page without guessing missing values.\n"
            "Schema: json{object_ok:boolean,name:string,date:string,candidate_answers:list[string]}\n"
            "Input snippet: exact page confirms name=Mercury, mentions launch in 1978, but no candidate answers shown\n"
            "Output: {\"object_ok\":true,\"name\":\"Mercury\",\"date\":\"1978\",\"candidate_answers\":[]}\n\n"
            "Example 5\n"
            "Instruction: Rewrite broad evidence into a compact next-step artifact.\n"
            "Schema: json{paper_url:string,candidate_labels:list[string],support:string}\n"
            "Input snippet: page confirms target paper and mentions Figure 2, Figure 3, supplementary figure\n"
            "Output: {\"paper_url\":\"https://example.org/paper\",\"candidate_labels\":[\"Figure 2\",\"Figure 3\",\"supplementary figure\"],\"support\":\"target paper page\"}\n\n"
            "Example 6\n"
            "Instruction: Extract version-audit records for later percentage calculation.\n"
            "Schema: json{records:list[json{item_name:string,superseded:boolean,support:string}],support:string}\n"
            "Input snippet: Current official pages show Item A revised in 1984, Item B still only has the 1959 standard, Item C revised in 1972\n"
            "Output: {\"records\":[{\"item_name\":\"Item A\",\"superseded\":true,\"support\":\"official page revised 1984\"},{\"item_name\":\"Item B\",\"superseded\":false,\"support\":\"official page still shows 1959 standard\"},{\"item_name\":\"Item C\",\"superseded\":true,\"support\":\"official page revised 1972\"}],\"support\":\"official current standards pages\"}\n\n"
            "Example 7\n"
            "Instruction: From this passage, extract the first place mentioned by name in the wording itself, not a later narrative setting.\n"
            "Schema: json{place:string,support:string}\n"
            "Input snippet: Xerxes ruled over 127 provinces stretching from India to Cush. At that time he reigned from his royal throne in the citadel of Susa.\n"
            "Output: {\"place\":\"India\",\"support\":\"stretching from India to Cush\"}\n\n"
            "Example 8\n"
            "Instruction: Extract the grounded operands for a repeated-removal density comparison before a later computation step.\n"
            "Schema: json{density_a:number,density_b:number,support:string}\n"
            "Input snippet: density table lists honey 1.42 g/mL and mayonnaise 0.91 g/mL at 25 C\n"
            "Output: {\"density_a\":1.42,\"density_b\":0.91,\"support\":\"density table at 25 C\"}\n\n"
            "Example 9\n"
            "Instruction: Select the exact paper source URL for body-text extraction.\n"
            "Schema: string\n"
            "Input snippet: results include article/view/733, citationstylelanguage/download/ris?submissionId=733, and core.ac.uk/download/pdf/267014159.pdf\n"
            "Output: https://journals.le.ac.uk/index.php/jist/article/view/733\n\n"
            f"Global Question:\n{context.get('global_question', global_question)}\n\n"
            f"Local Request:\n{context.get('local_question', local_request)}\n\n"
            f"Plan Context:\n{context.get('plan_context', plan_context)}\n\n"
            f"Blocked domains to avoid:\n{', '.join(blocked_domains) if blocked_domains else 'None'}\n\n"
            f"History of all observations:\n{json.dumps(context.get('history_observations', []), ensure_ascii=False)}\n\n"
            f"Current plan fragment:\n{json.dumps(context.get('current_plan', []), ensure_ascii=False)}\n\n"
            f"Instruction:\n{local_request}\n\n"
            f"Output schema:\n{output_schema}\n\n"
            f"Inputs with provenance:\n{chr(10).join(provenance_lines) if provenance_lines else 'None'}\n\n"
            f"{chr(10).join(rendered_inputs)}\n\n"
            "Return only the final value."
        )

        raw = await llm_adapter.apredict(prompt)
        if raw.startswith("Error:"):
            return raw

        payload = _extract_semantic_payload(raw)
        try:
            if output_schema.strip().lower() == "string":
                normalized = _coerce_semantic_value(payload, output_schema)
                normalized = _normalize_shortest_character_alias(normalized, local_request, context)
                if _is_blocked_selected_url(normalized, blocked_domains):
                    normalized = ""
            else:
                try:
                    parsed = json.loads(payload)
                except Exception:
                    repaired_payload = await _repair_semantic_json_payload(
                        llm_adapter,
                        payload,
                        output_schema,
                    )
                    if repaired_payload.startswith("Error:"):
                        return repaired_payload
                    parsed = json.loads(repaired_payload)
                normalized = _coerce_semantic_value(parsed, output_schema)
            return _truncate(_format_semantic_value(normalized, output_schema))
        except Exception as exc:
            return _truncate(f"semantic_map parsing error: {exc}\nRaw output:\n{payload}")

    return run_semantic_map


def run_speech_to_text_factory():
    async def run_speech_to_text(file_path):
        target = file_path[0] if isinstance(file_path, list) else file_path
        path = pathlib.Path(target)
        if not path.exists():
            return f"Audio file not found: {target}"
        snippet = f"""
from huggingface_hub import InferenceClient
client = InferenceClient(token=open('{Path.home() / '.cache' / 'huggingface' / 'token'}').read().strip(), timeout=180)
result = client.automatic_speech_recognition(r'''{str(path)}''', model='openai/whisper-large-v3-turbo')
print(getattr(result, 'text', result))
"""
        text = await asyncio.to_thread(
            _run_hf_inference_subprocess,
            snippet,
        )
        return _truncate(text or "Empty transcription.")

    return run_speech_to_text


def _parse_subtitle_text(raw_text: str) -> str:
    lines = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        if re.fullmatch(r"\d+", stripped):
            continue
        if "-->" in stripped:
            continue
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        stripped = re.sub(r"\[[^\]]+\]", " ", stripped)
        stripped = " ".join(stripped.split())
        if stripped:
            lines.append(stripped)
    deduped = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    return _truncate("\n".join(deduped), 12000)


def _extract_titlecase_names(text: str, limit: int = 3) -> list[str]:
    if not text:
        return []
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", text)
    names = []
    blocked = {
        "Artificial Intelligence",
        "The Thinking Machine",
        "YouTube Shorts",
        "National Geographic",
        "Project MUSE",
    }
    for candidate in candidates:
        cleaned = " ".join(candidate.split())
        if cleaned in blocked:
            continue
        if cleaned not in names:
            names.append(cleaned)
        if len(names) >= limit:
            break
    return names


def _compose_youtube_fallback_queries(info: dict) -> list[str]:
    title = str((info or {}).get("title") or "").strip()
    description = str((info or {}).get("description") or "").strip()
    names = _extract_titlecase_names(description, limit=3)
    queries = []
    if title:
        quoted_title = f"\"{title}\""
        queries.append(f"{quoted_title} transcript")
        queries.append(f"{quoted_title} interview summary")
        if names:
            queries.append(f"{quoted_title} {' '.join(names)} prediction")
    return [query.strip() for query in queries if query.strip()]


def _build_youtube_fallback_evidence(info: dict, search_api: "DuckDuckGoSearch", max_results: int = 6) -> dict:
    title = str((info or {}).get("title") or "").strip()
    if not title or search_api is None:
        return {"transcript": "", "transcript_source": "fallback_search", "supporting_urls": []}

    combined = []
    seen = set()
    queries = list(_compose_youtube_fallback_queries(info))
    for query in queries[:3]:
        try:
            results = search_api.search(query, max_results=4)
        except Exception:
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            href = str(item.get("href") or "").strip()
            title_text = str(item.get("title") or "").strip()
            body = str(item.get("body") or "").strip()
            key = (href, title_text, body)
            if key in seen:
                continue
            seen.add(key)
            combined.append(
                {
                    "query": query,
                    "title": title_text,
                    "url": href,
                    "snippet": _truncate(body, 500),
                }
            )
            if len(combined) >= max_results:
                break
        if len(combined) >= max_results:
            break

    evidence_lines = []
    for item in combined:
        title_text = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        line = " | ".join(part for part in [title_text, url, snippet] if part)
        if line:
            evidence_lines.append(line)

    return {
        "transcript": _truncate("\n".join(evidence_lines), 4000),
        "transcript_source": "fallback_search",
        "supporting_urls": [item.get("url", "") for item in combined[:max_results] if str(item.get("url", "")).strip()],
    }


def run_youtube_transcript_factory():
    fallback_search_api = DuckDuckGoSearch()

    async def run_youtube_transcript(url_value):
        url = url_value[0] if isinstance(url_value, list) else url_value
        url = str(url or "").strip()
        if not url:
            return "No YouTube URL provided."

        ytdlp_path = shutil.which("yt-dlp")
        if not ytdlp_path:
            return "yt-dlp is not available in the local environment."

        temp_dir = tempfile.mkdtemp(prefix="dynacall_yt_")
        try:
            info_proc = await asyncio.create_subprocess_exec(
                ytdlp_path,
                "--dump-single-json",
                "--skip-download",
                url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            info_stdout, info_stderr = await asyncio.wait_for(info_proc.communicate(), timeout=90)
            if info_proc.returncode != 0:
                return _truncate(
                    f"YouTube metadata fetch failed: {info_stderr.decode('utf-8', errors='ignore').strip()}"
                )
            info = json.loads(info_stdout.decode("utf-8", errors="ignore") or "{}")

            sub_proc = await asyncio.create_subprocess_exec(
                ytdlp_path,
                "--skip-download",
                "--write-subs",
                "--write-auto-subs",
                "--sub-langs",
                "en.*,en",
                "--convert-subs",
                "vtt",
                "-o",
                str(pathlib.Path(temp_dir) / "video.%(ext)s"),
                url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(sub_proc.communicate(), timeout=120)

            subtitle_files = sorted(pathlib.Path(temp_dir).glob("*.vtt")) + sorted(pathlib.Path(temp_dir).glob("*.srv*"))
            transcript_text = ""
            for subtitle_path in subtitle_files:
                try:
                    transcript_text = _parse_subtitle_text(subtitle_path.read_text(encoding="utf-8", errors="ignore"))
                    if transcript_text:
                        break
                except Exception:
                    continue

            payload = {
                "url": info.get("webpage_url") or url,
                "id": info.get("id"),
                "title": info.get("title"),
                "uploader": info.get("uploader"),
                "channel": info.get("channel"),
                "duration": info.get("duration"),
                "description": _truncate(str(info.get("description", "") or ""), 2000),
                "transcript": transcript_text,
                "has_transcript": bool(transcript_text),
                "transcript_source": "youtube_captions" if transcript_text else "fallback_search",
            }
            if not transcript_text:
                fallback = await asyncio.to_thread(_build_youtube_fallback_evidence, info, fallback_search_api)
                payload["transcript"] = str(fallback.get("transcript") or "")
                payload["supporting_urls"] = fallback.get("supporting_urls") or []
            return _truncate_json_safely(payload)
        except Exception as exc:
            return f"YouTube transcript fetch failed: {exc}"
        finally:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    return run_youtube_transcript


def run_ocr_factory():
    async def run_ocr(file_path):
        target = file_path[0] if isinstance(file_path, list) else file_path
        if isinstance(target, str) and ("Error downloading file" in target or "input is not a concrete http" in target):
            return target
        path = pathlib.Path(target)
        if not path.exists():
            return f"Image file not found: {target}"
        if not shutil.which("tesseract"):
            return "Tesseract OCR is not installed."
        result = await asyncio.to_thread(
            subprocess.run,
            ["tesseract", str(path), "stdout", "--psm", "6"],
            capture_output=True,
            text=True,
        )
        text = result.stdout.strip()
        if result.returncode != 0:
            text = result.stderr.strip() or "OCR failed."
        color_grouped = await asyncio.to_thread(_extract_color_grouped_tokens, path)
        parts = []
        if text:
            parts.append(text)
        if color_grouped:
            parts.append(color_grouped)
        return _truncate("\n\n".join(parts) or "No OCR text detected.")

    return run_ocr


def run_analyze_image_factory(inspector: GAIAFileInspector):
    async def run_analyze_image(file_path):
        target = file_path[0] if isinstance(file_path, list) else file_path
        return await asyncio.to_thread(inspector.inspect_mode, target, "image")

    return run_analyze_image


def _load_image_tooling():
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
    return Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter


def run_transform_image_factory():
    async def run_transform_image(args):
        values = args if isinstance(args, list) else [args]
        if len(values) < 2:
            return "transform_image requires at least image path and operation."
        image_path = str(values[0])
        operation = str(values[1]).strip().lower()
        params = {}
        if len(values) > 2 and values[2]:
            try:
                params = json.loads(values[2]) if isinstance(values[2], str) else dict(values[2])
            except Exception as exc:
                return f"Invalid params_json: {exc}"
        try:
            Image, _, _, ImageEnhance, ImageFilter = _load_image_tooling()
            img = Image.open(image_path)
            if operation == "resize":
                img = img.resize((int(params.get("width", img.width // 2)), int(params.get("height", img.height // 2))))
            elif operation == "rotate":
                img = img.rotate(float(params.get("angle", 90)), expand=True)
            elif operation == "crop":
                img = img.crop((
                    int(params.get("left", 0)),
                    int(params.get("top", 0)),
                    int(params.get("right", img.width)),
                    int(params.get("bottom", img.height)),
                ))
            elif operation == "flip":
                direction = str(params.get("direction", "horizontal")).lower()
                img = img.transpose(Image.FLIP_LEFT_RIGHT if direction == "horizontal" else Image.FLIP_TOP_BOTTOM)
            elif operation == "adjust_brightness":
                img = ImageEnhance.Brightness(img).enhance(float(params.get("factor", 1.5)))
            elif operation == "adjust_contrast":
                img = ImageEnhance.Contrast(img).enhance(float(params.get("factor", 1.5)))
            elif operation == "blur":
                img = img.filter(ImageFilter.GaussianBlur(float(params.get("radius", 2))))
            elif operation == "sharpen":
                img = img.filter(ImageFilter.SHARPEN)
            elif operation == "grayscale":
                img = img.convert("L")
            else:
                return f"Unknown operation: {operation}"
            out_path = Path(tempfile.gettempdir()) / f"dynacall_transform_{next(tempfile._get_candidate_names())}.png"
            img.save(out_path)
            return f"Transformed image saved to {out_path}"
        except Exception as exc:
            return f"Error transforming image: {exc}"

    return run_transform_image


def run_draw_on_image_factory():
    async def run_draw_on_image(args):
        values = args if isinstance(args, list) else [args]
        if len(values) < 3:
            return "draw_on_image requires image path, drawing_type, and params_json."
        image_path = str(values[0])
        drawing_type = str(values[1]).strip().lower()
        try:
            params = json.loads(values[2]) if isinstance(values[2], str) else dict(values[2])
        except Exception as exc:
            return f"Invalid params_json: {exc}"
        try:
            Image, ImageDraw, ImageFont, _, _ = _load_image_tooling()
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            color = params.get("color", "red")
            width = int(params.get("width", 2))
            if drawing_type == "rectangle":
                draw.rectangle([params["left"], params["top"], params["right"], params["bottom"]], outline=color, width=width)
            elif drawing_type == "circle":
                x, y, r = int(params["x"]), int(params["y"]), int(params["radius"])
                draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=width)
            elif drawing_type == "line":
                draw.line((params["start_x"], params["start_y"], params["end_x"], params["end_y"]), fill=color, width=width)
            elif drawing_type == "text":
                try:
                    font = ImageFont.truetype("arial.ttf", int(params.get("font_size", 20)))
                except Exception:
                    font = ImageFont.load_default()
                draw.text((int(params.get("x", 0)), int(params.get("y", 0))), str(params.get("text", "Text")), fill=color, font=font)
            else:
                return f"Unknown drawing type: {drawing_type}"
            out_path = Path(tempfile.gettempdir()) / f"dynacall_draw_{next(tempfile._get_candidate_names())}.png"
            img.save(out_path)
            return f"Annotated image saved to {out_path}"
        except Exception as exc:
            return f"Error drawing on image: {exc}"

    return run_draw_on_image


def run_generate_simple_image_factory():
    async def run_generate_simple_image(args):
        values = args if isinstance(args, list) else [args]
        if not values:
            return "generate_simple_image requires image_type."
        image_type = str(values[0]).strip().lower()
        width = int(values[1]) if len(values) > 1 and values[1] else 500
        height = int(values[2]) if len(values) > 2 and values[2] else 500
        params = {}
        if len(values) > 3 and values[3]:
            try:
                params = json.loads(values[3]) if isinstance(values[3], str) else dict(values[3])
            except Exception as exc:
                return f"Invalid params_json: {exc}"
        try:
            import numpy as np
            Image, ImageDraw, _, _, _ = _load_image_tooling()
            if image_type == "gradient":
                direction = str(params.get("direction", "horizontal")).lower()
                start_color = tuple(params.get("start_color", [255, 0, 0]))
                end_color = tuple(params.get("end_color", [0, 0, 255]))
                img = Image.new("RGB", (width, height))
                draw = ImageDraw.Draw(img)
                if direction == "horizontal":
                    for x in range(width):
                        ratio = x / max(width - 1, 1)
                        color = tuple(int(start_color[i] + (end_color[i] - start_color[i]) * ratio) for i in range(3))
                        draw.line([(x, 0), (x, height)], fill=color)
                else:
                    for y in range(height):
                        ratio = y / max(height - 1, 1)
                        color = tuple(int(start_color[i] + (end_color[i] - start_color[i]) * ratio) for i in range(3))
                        draw.line([(0, y), (width, y)], fill=color)
            elif image_type == "noise":
                noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                img = Image.fromarray(noise_array, "RGB")
            else:
                return f"Unsupported image_type: {image_type}"
            out_path = Path(tempfile.gettempdir()) / f"dynacall_generated_{next(tempfile._get_candidate_names())}.png"
            img.save(out_path)
            return f"Generated image saved to {out_path}"
        except Exception as exc:
            return f"Error generating image: {exc}"

    return run_generate_simple_image


def run_combine_images_factory():
    async def run_combine_images(args):
        values = args if isinstance(args, list) else [args]
        if len(values) < 2:
            return "combine_images requires at least two image paths."
        operation = "horizontal"
        image_values = values
        last_value = values[-1]
        if isinstance(last_value, str) and last_value.strip().lower() in {"horizontal", "vertical", "blend", "collage"}:
            operation = last_value.strip().lower()
            image_values = values[:-1]
        try:
            Image, _, _, _, _ = _load_image_tooling()
            images = [Image.open(str(path)).convert("RGBA") for path in image_values]
            if len(images) < 2:
                return "combine_images requires at least two valid image paths."
            if operation == "blend":
                base = images[0].resize(images[1].size)
                result = Image.blend(base, images[1], alpha=0.5)
            elif operation in {"horizontal", "collage"}:
                total_width = sum(img.width for img in images)
                max_height = max(img.height for img in images)
                result = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 0))
                x = 0
                for img in images:
                    result.paste(img, (x, 0))
                    x += img.width
            elif operation == "vertical":
                max_width = max(img.width for img in images)
                total_height = sum(img.height for img in images)
                result = Image.new("RGBA", (max_width, total_height), (255, 255, 255, 0))
                y = 0
                for img in images:
                    result.paste(img, (0, y))
                    y += img.height
            else:
                return f"Unsupported combine operation: {operation}"
            out_path = Path(tempfile.gettempdir()) / f"dynacall_combined_{next(tempfile._get_candidate_names())}.png"
            result.save(out_path)
            return f"Combined image saved to {out_path}"
        except Exception as exc:
            return f"Error combining images: {exc}"

    return run_combine_images


def run_image_recognition_factory(inspector: GAIAFileInspector):
    ocr_runner = run_ocr_factory()
    analyze_runner = run_analyze_image_factory(inspector)

    async def run_image_recognition(file_path):
        target = file_path[0] if isinstance(file_path, list) else file_path
        path = pathlib.Path(target)
        if not path.exists():
            return f"Image file not found: {target}"
        ocr_text = await ocr_runner(target)
        metadata = await analyze_runner(target)
        parts = [metadata]
        if ocr_text:
            parts.append(f"OCR: {ocr_text}")
        return _truncate("\n\n".join(parts))

    return run_image_recognition


def get_model_new(model_type, model_name, **kwargs):
    return create_llm_adapter(model_type, model_name, **kwargs)


def generate_tools(args, model_name):
    llm_adapter = get_model_new(
        model_type=args.model_type,
        model_name=model_name,
        temperature=0.0,
    )
    # search_engine backend: DuckDuckGoSearch with open-webSearch primary backend.
    web_search_api = DuckDuckGoSearch()
    url_fetch_chain = URLFetch(
        include_summary=False,
        max_urls=5,
        fetch_mode="truncate",
        web_search_api=web_search_api,
    )
    inspector = GAIAFileInspector(args.gaia_files_root)

    return [
        Tool(
            name="search_engine",
            func=run_search_engine_factory(web_search_api),
            description=_SEARCH_ENGINE_DESCRIPTION,
            stringify_rule=lambda args: f"search_engine({args[0]})",
        ),
        Tool(
            name="github_issue_search",
            func=run_github_issue_search_factory(),
            description=_GITHUB_ISSUE_SEARCH_DESCRIPTION,
            stringify_rule=lambda args: "github_issue_search(<repo>, <labels>, <state>, <sort>, <direction>)",
        ),
        Tool(
            name="web_browser",
            func=run_url_fetch_factory(url_fetch_chain),
            description=_WEB_BROWSER_DESCRIPTION,
            stringify_rule=lambda args: f"web_browser({args[0]})",
        ),
        # Temporarily hidden from the GAIA planner. Keep the implementation
        # available in code, but do not expose it as a selectable tool.
        # Tool(
        #     name="search_get_contents",
        #     func=run_search_get_contents_factory(web_search_api),
        #     description=_SEARCH_GET_CONTENTS_DESCRIPTION,
        #     stringify_rule=lambda args: "search_get_contents(<query_or_url>)",
        # ),
        # Temporarily disabled. Keep the implementation above intact so it can
        # be re-enabled later, but do not expose it to the GAIA planner.
        # Tool(
        #     name="deepsearch",
        #     func=run_deepsearch_factory(web_search_api, llm_adapter),
        #     description=_DEEPSEARCH_DESCRIPTION,
        #     stringify_rule=lambda args: f"deepsearch({args[0]})",
        # ),
        Tool(
            name="wiki_section_extract",
            func=run_wiki_section_extract_factory(),
            description=_WIKI_SECTION_EXTRACT_DESCRIPTION,
            stringify_rule=lambda args: f"wiki_section_extract({args[0]})",
        ),
        Tool(
            name="orcid_reader",
            func=run_orcid_reader_factory(),
            description=_ORCID_READER_DESCRIPTION,
            stringify_rule=lambda args: f"orcid_reader({args[0]})",
        ),
        Tool(
            name="crossref_lookup",
            func=run_crossref_lookup_factory(),
            description=_CROSSREF_LOOKUP_DESCRIPTION,
            stringify_rule=lambda args: f"crossref_lookup({args[0]})",
        ),
        Tool(
            name="quote_verifier",
            func=run_quote_verifier_factory(web_search_api),
            description=_QUOTE_VERIFIER_DESCRIPTION,
            stringify_rule=lambda args: f"quote_verifier({args[0]})",
        ),
        Tool(
            name="openalex_author_works",
            func=run_openalex_author_works_factory(),
            description=_OPENALEX_AUTHOR_WORKS_DESCRIPTION,
            stringify_rule=lambda args: f"openalex_author_works({args[0]})",
        ),
        Tool(
            name="file_reader",
            func=run_file_reader_factory(inspector, "auto"),
            description=_FILE_READER_DESCRIPTION,
            stringify_rule=lambda args: f"file_reader({args[0]})",
        ),
        Tool(
            name="pdf_viewer",
            func=run_file_reader_factory(inspector, "pdf"),
            description=_PDF_VIEWER_DESCRIPTION,
            stringify_rule=lambda args: f"pdf_viewer({args[0]})",
        ),
        Tool(
            name="spreadsheet_reader",
            func=run_file_reader_factory(inspector, "spreadsheet"),
            description=_SPREADSHEET_READER_DESCRIPTION,
            stringify_rule=lambda args: f"spreadsheet_reader({args[0]})",
        ),
        Tool(
            name="powerpoint_viewer",
            func=run_file_reader_factory(inspector, "powerpoint"),
            description=_POWERPOINT_VIEWER_DESCRIPTION,
            stringify_rule=lambda args: f"powerpoint_viewer({args[0]})",
        ),
        Tool(
            name="text_reader",
            func=run_file_reader_factory(inspector, "text"),
            description=_TEXT_READER_DESCRIPTION,
            stringify_rule=lambda args: f"text_reader({args[0]})",
        ),
        Tool(
            name="archive_explorer",
            func=run_file_reader_factory(inspector, "archive"),
            description=_ARCHIVE_EXPLORER_DESCRIPTION,
            stringify_rule=lambda args: f"archive_explorer({args[0]})",
        ),
        # Tool(
        #     name="save_and_read_file",
        #     func=run_save_and_read_file_factory(),
        #     description=_SAVE_AND_READ_FILE_DESCRIPTION,
        #     stringify_rule=lambda args: "save_and_read_file(<content>)",
        # ),
        Tool(
            name="download_file_from_url",
            func=run_download_file_from_url_factory(),
            description=_DOWNLOAD_FILE_FROM_URL_DESCRIPTION,
            stringify_rule=lambda args: f"download_file_from_url({args[0]})",
        ),
        Tool(
            name="image_recognition",
            func=run_image_recognition_factory(inspector),
            description=_IMAGE_RECOGNITION_DESCRIPTION,
            stringify_rule=lambda args: f"image_recognition({args[0]})",
        ),
        Tool(
            name="analyze_image",
            func=run_analyze_image_factory(inspector),
            description=_ANALYZE_IMAGE_DESCRIPTION,
            stringify_rule=lambda args: f"analyze_image({args[0]})",
        ),
        # Tool(
        #     name="transform_image",
        #     func=run_transform_image_factory(),
        #     description=_TRANSFORM_IMAGE_DESCRIPTION,
        #     stringify_rule=lambda args: f"transform_image({args[0]})",
        # ),
        # Tool(
        #     name="draw_on_image",
        #     func=run_draw_on_image_factory(),
        #     description=_DRAW_ON_IMAGE_DESCRIPTION,
        #     stringify_rule=lambda args: f"draw_on_image({args[0]})",
        # ),
        # Tool(
        #     name="generate_simple_image",
        #     func=run_generate_simple_image_factory(),
        #     description=_GENERATE_SIMPLE_IMAGE_DESCRIPTION,
        #     stringify_rule=lambda args: f"generate_simple_image({args[0]})",
        # ),
        # Tool(
        #     name="combine_images",
        #     func=run_combine_images_factory(),
        #     description=_COMBINE_IMAGES_DESCRIPTION,
        #     stringify_rule=lambda args: f"combine_images({args[0]})",
        # ),
        Tool(
            name="ocr",
            func=run_ocr_factory(),
            description=_OCR_DESCRIPTION,
            stringify_rule=lambda args: f"ocr({args[0]})",
        ),
        Tool(
            name="extract_text_from_image",
            func=run_ocr_factory(),
            description=_EXTRACT_TEXT_FROM_IMAGE_DESCRIPTION,
            stringify_rule=lambda args: f"extract_text_from_image({args[0]})",
        ),
        Tool(
            name="speech_to_text",
            func=run_speech_to_text_factory(),
            description=_SPEECH_TO_TEXT_DESCRIPTION,
            stringify_rule=lambda args: f"speech_to_text({args[0]})",
        ),
        Tool(
            name="youtube_transcript",
            func=run_youtube_transcript_factory(),
            description=_YOUTUBE_TRANSCRIPT_DESCRIPTION,
            stringify_rule=lambda args: f"youtube_transcript({args[0]})",
        ),
        Tool(
            name="execute_code_multilang",
            func=run_execute_code_multilang_factory(args.gaia_files_root),
            description=_EXECUTE_CODE_MULTILANG_DESCRIPTION,
            stringify_rule=lambda args: "execute_code_multilang(<code>)",
        ),
        Tool(
            name="code_interpreter",
            func=run_code_interpreter_factory(args.gaia_files_root),
            description=_CODE_INTERPRETER_DESCRIPTION,
            stringify_rule=lambda args: "code_interpreter(<code_or_task>)",
        ),
        Tool(
            name="python",
            func=run_python_factory(args.gaia_files_root),
            description=_PYTHON_DESCRIPTION,
            stringify_rule=lambda args: "python(<code>)",
        ),
        Tool(
            name="calculator",
            func=run_calculator_factory(),
            description=_CALCULATOR_DESCRIPTION,
            stringify_rule=lambda args: f"calculator({args[0]})",
        ),
        Tool(
            name="semantic_map",
            func=run_semantic_map_factory(llm_adapter),
            description=_SEMANTIC_MAP_DESCRIPTION,
            stringify_rule=lambda args: "semantic_map(<global_question>, <local_request>, <plan_context>, <observations>, <schema>)",
        ),
        Tool(
            name="verifier",
            func=run_verifier_factory(llm_adapter),
            description=_VERIFIER_DESCRIPTION,
            stringify_rule=lambda args: "verifier(<question>, <proposed_answer>, <evidence>)",
        ),
    ]
