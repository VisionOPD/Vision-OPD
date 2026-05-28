import re
import string
from decimal import Decimal, InvalidOperation

try:
    from mathruler.grader import grade_answer
except Exception:
    grade_answer = None


def extract_answer(model_answer_raw: str) -> str:
    if not isinstance(model_answer_raw, str):
        return ""
    start = model_answer_raw.find("<answer>")
    end = model_answer_raw.find("</answer>")
    if start != -1 and end != -1 and end > start:
        return model_answer_raw[start + len("<answer>") : end].strip()
    if "Answer:" in model_answer_raw:
        return model_answer_raw[model_answer_raw.find("Answer:") + len("Answer:") :].strip()
    return model_answer_raw.strip()


def extract_first_option(text: str) -> str:
    if not text:
        return ""

    match = re.search(r"\(([A-Z])\)", text)
    if match:
        return match.group(1)

    match = re.search(r"([A-Z])[\.\)\s]", text)
    if match:
        return match.group(1)

    match = re.search(r"([A-Z])", text)
    if match:
        return match.group(1)

    return ""


def extract_mcq_option(answer: str) -> str:
    if not isinstance(answer, str) or not answer:
        return ""
    text = answer.strip()
    match = re.match(r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))", text)
    if match:
        return match.group(1)
    return ""


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    without_punc = "".join(ch for ch in lowered if ch not in string.punctuation)
    return " ".join(without_punc.split())


def maybe_parse_number(text: str) -> Decimal | None:
    if not isinstance(text, str):
        return None
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text.replace("−", "-"))
    if not match:
        return None
    try:
        return Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return None


def compute_score(solution_str: str, ground_truth: str, extra_info=None):
    extra_info = extra_info or {}
    question_type = str(extra_info.get("question_type") or "").strip().lower()
    extracted_answer = extract_answer(solution_str)

    if question_type == "mcq":
        pred = extract_first_option(extracted_answer)
        gt = extract_mcq_option(str(ground_truth))
        is_correct = bool(pred and gt and pred == gt)
        return {
            "score": float(is_correct),
            "acc": float(is_correct),
            "pred": pred or extracted_answer,
            "extracted_answer": extracted_answer,
            "judge_source": "mcq_rule",
        }

    if grade_answer is not None:
        try:
            is_correct = bool(grade_answer(ground_truth, extracted_answer))
            return {
                "score": float(is_correct),
                "acc": float(is_correct),
                "pred": extracted_answer,
                "extracted_answer": extracted_answer,
                "judge_source": "mathruler",
            }
        except Exception:
            pass

    pred_num = maybe_parse_number(extracted_answer)
    gt_num = maybe_parse_number(str(ground_truth))
    if pred_num is not None and gt_num is not None:
        is_correct = pred_num == gt_num
        return {
            "score": float(is_correct),
            "acc": float(is_correct),
            "pred": str(pred_num),
            "extracted_answer": extracted_answer,
            "judge_source": "numeric_rule",
        }

    is_correct = normalize_text(extracted_answer) == normalize_text(str(ground_truth))
    return {
        "score": float(is_correct),
        "acc": float(is_correct),
        "pred": extracted_answer,
        "extracted_answer": extracted_answer,
        "judge_source": "normalized_em",
    }
