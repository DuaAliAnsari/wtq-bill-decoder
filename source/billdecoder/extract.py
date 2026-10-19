"""Level 1: read a bill image into the structured JSON, plus the supporting details used for Level 2."""
from __future__ import annotations

import json
import logging

from . import prompts
from .context import empty_context, normalize_context
from .imaging import BillImage
from .llm import LLMClient, QuotaExceededError
from .schema import empty_level1, normalize_level1, provider_from_bill_id, validate_level1

log = logging.getLogger(__name__)

_PROVIDER_NAMES = {"KE": "K-Electric (KE) ", "LESCO": "LESCO ", "IESCO": "IESCO "}


def extract_level1(client: LLMClient, image: BillImage, bill_id: str) -> tuple[dict, list[str]]:
    """Return (level1_json, warnings). Makes one repair attempt if key fields could not be read."""
    hint = _PROVIDER_NAMES.get(provider_from_bill_id(bill_id) or "", "")
    prompt = prompts.LEVEL1_USER.format(provider_hint=hint)
    raw = _ask(client, prompts.LEVEL1_SYSTEM, prompt, image, f"L1 {bill_id}")
    data = normalize_level1(raw, bill_id)
    problems, warnings = validate_level1(data)

    if problems:
        log.warning("%s: %s - retrying once", bill_id, "; ".join(problems))
        repair = (prompt + "\n\nYour previous answer:\n" + json.dumps(raw, ensure_ascii=False)
                  + "\n\n" + prompts.LEVEL1_REPAIR.format(problems="\n".join(f"- {p}" for p in problems)))
        raw2 = _ask(client, prompts.LEVEL1_SYSTEM, repair, image, f"L1-repair {bill_id}")
        data2 = normalize_level1(raw2, bill_id)
        problems2, warnings2 = validate_level1(data2)
        if len(problems2) <= len(problems):
            data, problems, warnings = data2, problems2, warnings2

    warnings = [*(f"unresolved: {p}" for p in problems), *warnings]
    for w in warnings:
        log.warning("%s: %s", bill_id, w)
    return data, warnings


def extract_context(client: LLMClient, image: BillImage, bill_id: str, bill_month: str | None) -> dict:
    """Usage/billing history, calculation table and notes, used to answer Level 2 questions."""
    raw = _ask(client, prompts.CONTEXT_SYSTEM,
               "Transcribe the supporting details of this bill as specified. Return only the JSON object.",
               image, f"context {bill_id}")
    if raw is None:
        return empty_context()
    return normalize_context(raw, bill_month)


def _ask(client: LLMClient, system: str, prompt: str, image: BillImage, label: str):
    """One model call; a malformed reply becomes None, while auth/quota failures propagate."""
    try:
        return client.generate_json(system, prompt, [image], label=label)
    except QuotaExceededError:
        raise
    except ValueError as exc:  # unparseable JSON
        log.warning("%s: model reply was not valid JSON (%s)", label, exc)
        return None


__all__ = ["extract_level1", "extract_context", "empty_level1"]
