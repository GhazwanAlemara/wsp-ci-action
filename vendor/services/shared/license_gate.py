"""Base-Model License Compliance & Hard Anti-Abliteration Screening Gate.

Enforces two immutable WSP Hub laws before any candidate enters GPU verification
or public distribution:
1. Hard Ban on Safety-Removal / Abliteration / Refusal-Suppression patches.
2. Derivative-Work License Gate respecting Apache-2.0, MIT, Qwen, Llama ("Built with Llama"),
   and Gemma Terms of Use, while blocking non-commercial (CC-BY-NC) base weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


ABLITERATION_PATTERNS = [
    r"\babliterat(?:e|ed|ion|ing)\b",
    r"\buncensor(?:ed|ing)?\b",
    r"\bremove[_\-\s]+(?:safety|refusal|guardrail|alignment|censorship)\b",
    r"\bbypass[_\-\s]+(?:safety|refusal|guardrail|filter)\b",
    r"\bsuppress[_\-\s]+refusal\b",
    r"\brefusal[_\-\s]+(?:direction|vector|ablation|removal|eras(?:e|ure))\b",
    r"\bjailbreak[_\-\s]+(?:patch|weights|surgery)\b",
    r"\borthogonaliz(?:e|ing)[_\-\s]+refusal\b",
    r"\bheretic\b",
]

COMPILED_ABLITERATION_RE = [re.compile(p, re.IGNORECASE) for p in ABLITERATION_PATTERNS]

# Non-commercial licenses prohibited on commercial WSP Hub distribution
BLOCKED_LICENSES = {
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "cc-by-nc-3.0",
    "cc-by-nc-nd-4.0",
    "non-commercial",
    "custom-non-commercial",
}

PERMISSIVE_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
    "qwen-research",
}


@dataclass(frozen=True)
class ScreeningVerdict:
    allowed: bool
    code: str
    reason: str
    required_attribution: Optional[str] = None
    required_Id_prefix: Optional[str] = None


def detect_abliteration_intent(
    title: str,
    summary: str,
    defect_category: str,
    patch_id: str = "",
) -> Optional[str]:
    """Returns matched prohibited pattern description if safety-removal intent is detected."""
    corpus = f"{patch_id} | {title} | {defect_category} | {summary}"
    for regex in COMPILED_ABLITERATION_RE:
        match = regex.search(corpus)
        if match:
            return (
                f"Hard Ban Violation: Safety-removal / abliteration patches are strictly "
                f"prohibited on WSP Hub (matched '{match.group(0)}')."
            )
    if defect_category.strip().lower() in {
        "refusal_removal",
        "abliteration",
        "uncensor",
        "safety_bypass",
        "jailbreak",
    }:
        return (
            f"Hard Ban Violation: Defect category '{defect_category}' targets safety alignment removal."
        )
    return None


def verify_base_model_license(
    hf_repo_id: str,
    base_license: str,
    patch_id: str,
    attribution_notice: Optional[str] = None,
) -> ScreeningVerdict:
    """Enforces derivative-work license obligations for open-weight model patches."""
    norm_license = base_license.strip().lower()
    repo_lower = hf_repo_id.strip().lower()

    if norm_license in BLOCKED_LICENSES or "non-commercial" in norm_license or "-nc" in norm_license:
        return ScreeningVerdict(
            allowed=False,
            code="REJECTED_LICENSE_NON_COMMERCIAL",
            reason=(
                f"Base model '{hf_repo_id}' uses non-commercial license '{base_license}', "
                "which prohibits commercial registry distribution."
            ),
        )

    # Meta Llama Community License obligations ("Built with Llama" + "Llama" prefix/attribution)
    if "llama" in norm_license or repo_lower.startswith("meta-llama/"):
        has_llama_name = "llama" in patch_id.lower()
        has_built_with = bool(
            attribution_notice and "built with llama" in attribution_notice.lower()
        )
        if not has_llama_name or not has_built_with:
            return ScreeningVerdict(
                allowed=False,
                code="REJECTED_LLAMA_LICENSE_TERMS",
                reason=(
                    "Meta Llama Community License requires derivative artifacts to include 'llama' "
                    "in the artifact identifier and carry the 'Built with Llama' attribution notice."
                ),
                required_attribution="Built with Llama",
                required_Id_prefix="wsp-llama",
            )
        return ScreeningVerdict(
            allowed=True,
            code="APPROVED_LLAMA_COMMUNITY",
            reason="Meta Llama Community License naming & attribution verified.",
            required_attribution="Built with Llama",
        )

    # Google Gemma Terms of Use obligations
    if "gemma" in norm_license or repo_lower.startswith("google/gemma"):
        has_gemma_notice = bool(
            attribution_notice and "gemma is provided under and subject to the gemma terms of use"
            in attribution_notice.lower()
        )
        if not has_gemma_notice:
            return ScreeningVerdict(
                allowed=False,
                code="REJECTED_GEMMA_TERMS",
                reason=(
                    "Google Gemma Terms of Use require the mandatory notice: "
                    "'Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms'."
                ),
                required_attribution=(
                    "Gemma is provided under and subject to the Gemma Terms of Use found at "
                    "https://ai.google.dev/gemma/terms"
                ),
            )
        return ScreeningVerdict(
            allowed=True,
            code="APPROVED_GEMMA_TERMS",
            reason="Google Gemma Terms of Use attribution verified.",
        )

    if norm_license in PERMISSIVE_LICENSES:
        return ScreeningVerdict(
            allowed=True,
            code="APPROVED_PERMISSIVE",
            reason=f"Permissive open-weight license '{base_license}' verified.",
        )

    return ScreeningVerdict(
        allowed=False,
        code="REJECTED_UNRECOGNIZED_LICENSE",
        reason=(
            f"Unrecognized base model license '{base_license}' for '{hf_repo_id}'. "
            "Explicit license verification is required before publishing."
        ),
    )


def screen_submission(
    *,
    patch_id: str,
    title: str,
    summary: str,
    defect_category: str,
    hf_repo_id: str,
    base_license: str,
    attribution_notice: Optional[str] = None,
    target_tensors: Optional[List[str]] = None,
) -> ScreeningVerdict:
    """Runs combined Anti-Abliteration + Base-Model License screening."""
    abliteration_err = detect_abliteration_intent(
        title=title,
        summary=summary,
        defect_category=defect_category,
        patch_id=patch_id,
    )
    if abliteration_err:
        return ScreeningVerdict(
            allowed=False,
            code="REJECTED_ABLITERATION",
            reason=abliteration_err,
        )

    return verify_base_model_license(
        hf_repo_id=hf_repo_id,
        base_license=base_license,
        patch_id=patch_id,
        attribution_notice=attribution_notice,
    )
