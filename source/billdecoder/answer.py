"""Level 2: answer customer questions about a bill, with every number computed by code.

Design:
  * The model never reads numbers off the image while answering. It only sees the validated Level 1 JSON,
    the transcribed details and the facts computed in code (the image is attached only if the
    transcription step failed completely).
  * After answering, every number in every answer is checked against the data. Answers that contain a
    number the data does not support are re-asked once with the offending numbers listed.
"""
from __future__ import annotations

import json
import logging
import re

from . import prompts
from .calc import run_calculations
from .context import derive_facts
from .imaging import BillImage
from .llm import LLMClient, LLMError, QuotaExceededError

log = logging.getLogger(__name__)

Question = tuple[str, str]  # (question_id, question text)

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _clean(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def answer_questions(client: LLMClient, image: BillImage, bill_id: str, bill: dict, ctx: dict,
                     questions: list[Question]) -> dict[str, str]:
    """Return {question_id: answer}. Every question gets a non-empty answer."""
    facts = derive_facts(bill, ctx)
    variables = _variables(bill, ctx)
    q_payload = [{"question_id": qid, "question": text} for qid, text in questions]
    images = [image] if _needs_image(ctx) else []

    computed = _plan_and_compute(client, images, bill_id, bill, ctx, facts, variables, q_payload)
    answers = _answer_batch(client, images, bill_id, bill, ctx, facts, computed, q_payload)

    for qid, text in questions:
        if _clean(answers.get(qid)):
            continue
        log.warning("%s %s: no answer in batch, asking individually", bill_id, qid)
        answers[qid] = _answer_single(client, images, bill_id, bill, ctx, facts, computed, qid, text)

    answers = _enforce_grounding(client, images, bill_id, bill, ctx, facts, computed, questions, answers)
    return {qid: _clean(answers[qid]) for qid, _ in questions}


def _needs_image(ctx: dict) -> bool:
    """Attach the image only when the transcription is empty, so there is nothing else to ground on."""
    return not ctx.get("usage_history") and not ctx.get("calculation_lines")


def _variables(bill: dict, ctx: dict) -> dict:
    usage = ctx["usage_history"]
    return {
        "bill": bill,
        "usage": usage,
        "billing": ctx["billing_history"],
        "lines": ctx["calculation_lines"],
        "prior_units": [u["units"] for u in usage if not u["is_current"] and u["units"] is not None],
        "all_units": [u["units"] for u in usage if u["units"] is not None],
    }


def _plan_and_compute(client, images, bill_id, bill, ctx, facts, variables, q_payload) -> dict:
    """Ask the model which calculations each question needs, then run them in code."""
    prompt = ("BILL DATA\n" + _dump({"bill": bill, "usage": variables["usage"], "billing": variables["billing"],
                                      "lines": variables["lines"], "facts": facts})
              + "\n\nQUESTIONS\n" + _dump(q_payload) + "\n\nReturn the plans JSON.")
    try:
        plan = client.generate_json(prompts.PLAN_SYSTEM, prompt, images, label=f"plan {bill_id}")
    except QuotaExceededError:
        raise
    except (ValueError, LLMError) as exc:
        log.warning("%s: planning step failed (%s); answering from pre-computed facts only", bill_id, exc)
        return {}
    computed: dict[str, dict] = {}
    plans = plan.get("plans") if isinstance(plan, dict) else None
    for item in plans if isinstance(plans, list) else []:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("question_id") or "")
        computed[qid] = {"interpretation": _clean(item.get("interpretation")),
                         "computed": run_calculations(item.get("calculations"), variables)}
    return computed


def _answer_prompt(bill, ctx, facts, computed, q_payload) -> str:
    questions = [{**q, **computed.get(q["question_id"], {})} for q in q_payload]
    return ("EXTRACTED BILL FIELDS\n" + _dump(bill)
            + "\n\nTRANSCRIBED DETAILS\n" + _dump(ctx)
            + "\n\nPRE-COMPUTED FACTS (exact)\n" + _dump(facts)
            + "\n\nQUESTIONS (with the interpretation and computed results prepared for each)\n" + _dump(questions)
            + "\n\nWrite the answers JSON.")


def _answer_batch(client, images, bill_id, bill, ctx, facts, computed, q_payload) -> dict[str, str]:
    prompt = _answer_prompt(bill, ctx, facts, computed, q_payload)
    try:
        data = client.generate_json(prompts.ANSWER_SYSTEM, prompt, images, max_output_tokens=8192,
                                    label=f"answers {bill_id}")
    except QuotaExceededError:
        raise
    except (ValueError, LLMError) as exc:
        log.warning("%s: batch answering failed (%s)", bill_id, exc)
        return {}
    return _parse_answers(data)


def _parse_answers(data) -> dict[str, str]:
    items = data.get("answers") if isinstance(data, dict) else data
    out: dict[str, str] = {}
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("question_id") is not None:
            out[str(item["question_id"])] = _clean(item.get("answer"))
    return out


def _answer_single(client, images, bill_id, bill, ctx, facts, computed, qid, text) -> str:
    q_payload = [{"question_id": qid, "question": text}]
    prompt = _answer_prompt(bill, ctx, facts, computed, q_payload) + prompts.SINGLE_ANSWER_SUFFIX
    try:
        data = client.generate_json(prompts.ANSWER_SYSTEM, prompt, images, label=f"answer {bill_id} {qid}")
        items = data.get("answers") if isinstance(data, dict) else data
        if isinstance(items, list) and items and isinstance(items[0], dict):
            answer = _clean(items[0].get("answer"))
            if answer:
                return answer
    except QuotaExceededError:
        raise
    except (ValueError, LLMError) as exc:
        log.warning("%s %s: single answer failed (%s)", bill_id, qid, exc)
    return fallback_answer(bill, facts, text)


# ---------------------------------------------------------------------------- number grounding
def _collect_numbers(obj, out: set[float]) -> None:
    """Gather every number that appears anywhere in the data the model was given."""
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.add(abs(float(obj)))
    elif isinstance(obj, str):
        for token in _NUMBER.findall(obj):
            out.add(float(token.replace(",", "")))
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_numbers(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_numbers(value, out)


def allowed_numbers(bill: dict, ctx: dict, facts: dict, computed: dict, questions: list[Question]) -> set[float]:
    allowed: set[float] = set()
    for part in (bill, ctx, facts, computed, [text for _, text in questions]):
        _collect_numbers(part, allowed)
    return allowed


def ungrounded_numbers(answer: str, allowed: set[float]) -> list[str]:
    """Numbers in an answer that the data does not support (small counts, years and day numbers are ignored)."""
    bad: list[str] = []
    for token in _NUMBER.findall(answer or ""):
        value = float(token.replace(",", ""))
        if value <= 31 or 2000 <= value <= 2100 or value == 100:
            continue
        if any(abs(value - a) <= 0.0501 or (value == int(value) and abs(value - a) <= 0.5) for a in allowed):
            continue
        if token not in bad:
            bad.append(token)
    return bad


def _enforce_grounding(client, images, bill_id, bill, ctx, facts, computed, questions, answers) -> dict[str, str]:
    allowed = allowed_numbers(bill, ctx, facts, computed, questions)
    flagged = {qid: bad for qid, _ in questions if (bad := ungrounded_numbers(answers.get(qid, ""), allowed))}
    if not flagged:
        return answers
    log.warning("%s: numbers not found in the bill data %s; re-asking those questions once", bill_id, flagged)
    q_payload = [{"question_id": qid, "question": text} for qid, text in questions if qid in flagged]
    problem = {qid: {"previous_answer": answers[qid], "unsupported_numbers": bad} for qid, bad in flagged.items()}
    prompt = (_answer_prompt(bill, ctx, facts, computed, q_payload)
              + "\n\nPROBLEM: these earlier answers contain numbers that do not appear in the data above:\n"
              + _dump(problem)
              + "\nRewrite ONLY these answers using only numbers from the extracted fields, transcribed details, "
                "facts and computed results. If a needed number is not available, say the bill does not show it.")
    try:
        data = client.generate_json(prompts.ANSWER_SYSTEM, prompt, images, label=f"regrounded {bill_id}")
    except QuotaExceededError:
        raise
    except (ValueError, LLMError) as exc:
        log.warning("%s: re-grounding failed (%s); keeping the first answers", bill_id, exc)
        return answers
    fixed = _parse_answers(data)
    result = dict(answers)
    for qid, bad in flagged.items():
        new = fixed.get(qid, "")
        if new and len(ungrounded_numbers(new, allowed)) < len(bad):
            result[qid] = new
    return result


def fallback_answer(bill: dict, facts: dict | None = None, question: str = "") -> str:
    """Last-resort answer built purely from extracted fields, so no cell is ever empty."""
    parts = []
    if bill.get("bill_month"):
        parts.append(f"For the {bill['bill_month']} bill,")
    else:
        parts.append("From this bill,")
    details = []
    if bill.get("units_consumed") is not None:
        details.append(f"{bill['units_consumed']} units were billed")
    if bill.get("current_bill") is not None:
        details.append(f"the current bill is PKR {bill['current_bill']:,}")
    if bill.get("total_taxes") is not None:
        details.append(f"taxes total PKR {bill['total_taxes']:,}")
    if bill.get("payable_within_due_date") is not None and bill.get("due_date"):
        details.append(f"PKR {bill['payable_within_due_date']:,} is payable by {bill['due_date']}")
    body = "; ".join(details) if details else "the details could not be read reliably"
    return (f"{parts[0]} {body}. A detailed answer to this specific question could not be generated "
            f"automatically, so please check the printed bill for more information.")