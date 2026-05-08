import re
import string
import time
import traceback
from typing import Optional, Union


def extract_final_answer_text(text):
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return text

    final_answer_match = re.search(r"FINAL ANSWER:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if final_answer_match:
        text = final_answer_match.group(1).strip()

    finish_match = re.search(r"Finish\((.*)\)", text, re.DOTALL)
    if finish_match:
        text = finish_match.group(1).strip()

    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def normalize_answer(s):
    s = extract_final_answer_text(s)
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
def normalize_gaia_answer(s):
    s = extract_final_answer_text(s)
    if s is None:
        return None
    return " ".join(str(s).strip().lower().split())


def compare_answer_parallelqa(answer, label):
    """Legacy comparator used by ParallelQA-style benchmarks.

    - Numeric labels allow a 10% margin.
    - Text labels use aggressive normalization.
    """
    if answer is None:
        return False

    answer = extract_final_answer_text(answer)
    label = extract_final_answer_text(label)

    if is_number(label):
        label = float(label)
        try:
            answer = float(answer)
        except:
            return False
        return answer > label * 0.9 and answer < label * 1.1

    label = normalize_answer(label)
    answer = normalize_answer(answer)
    return answer == label


def compare_answer_gaia(answer, label):
    """GAIA-specific comparator.

    GAIA answers are typically exact strings, identifiers, zip codes, dates,
    or exact computed values. We therefore avoid the ParallelQA 10% numeric
    tolerance and use exact matching after only light normalization.
    """
    if answer is None:
        return False

    answer = extract_final_answer_text(answer)
    label = extract_final_answer_text(label)

    if answer is None or label is None:
        return False

    if is_number(label) and is_number(answer):
        try:
            return float(answer) == float(label)
        except:
            return False

    return normalize_gaia_answer(answer) == normalize_gaia_answer(label)


def compare_answer(answer, label, benchmark_name: Optional[str] = None):
    if benchmark_name in {"gaia", "bfcl_ws", "bfcl_ws_llmcompiler", "bfcl_ws_wotbranch", "bfcl_ws_wotsm", "bfcl_ws_react"}:
        return compare_answer_gaia(answer, label)
    return compare_answer_parallelqa(answer, label)


def compare_answer_for_benchmark(answer, label, benchmark_name: Optional[str] = None):
    if benchmark_name in {"gaia", "gaia_llmcompiler", "bfcl_ws", "bfcl_ws_llmcompiler", "bfcl_ws_wotbranch", "bfcl_ws_wotsm", "bfcl_ws_react"}:
        return compare_answer_gaia(answer, label)
    return compare_answer_parallelqa(answer, label)


def run_and_time(func, *args, **kwargs):
    """helper function to run and time a function.
    Since function can error, we catch the error and return "ERROR" as the result
    """
    start = time.time()
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        print("Error", e)
        traceback.print_exc()
        result = "ERROR"
    end = time.time()
    return result, end - start
async def arun_and_time(func, *args, **kwargs):
    """helper function to run and time a function.
    Since function can error, we catch the error and return "ERROR" as the result
    """
    start = time.time()
    try:
        result = await func(*args, **kwargs)
    except Exception as e:
        print("Error", e)
        traceback.print_exc()
        result = "ERROR"
    end = time.time()
    return result, end - start
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
