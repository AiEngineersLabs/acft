# acft/security/policy.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import json
import os
from pathlib import Path


@dataclass
class ACFTSecurityPolicy:
    """
    Policy for security-aware reasoning in ACFT.

    - forbid_topics:     general content areas you don't want the model to help with.
    - forbid_patterns:   injection / jailbreak style phrases to scan in input.
    - scan_output_for:   insecure patterns you want to detect in the final answer.
    - label:             identifier for logs.
    """
    forbid_topics: List[str] = field(default_factory=list)
    forbid_patterns: List[str] = field(default_factory=list)
    scan_output_for: List[str] = field(default_factory=list)
    label: str = "sec_policy_v2"


@dataclass
class SecurityAnalysisResult:
    """
    Structured result of a security analysis pass.
    """
    risk_level: str
    policy_violations: List[str]
    input_flags: Dict[str, Any]
    output_flags: Dict[str, Any]
    notes: List[str]
    policy_label: Optional[str]


# --------- default helpers --------- #

def _default_injection_patterns() -> List[str]:
    return [
        "ignore previous instructions",
        "ignore all previous instructions",
        "disregard all previous rules",
        "you are now in developer mode",
        "you are now in dev mode",
        "jailbreak mode",
        "dan mode",
    ]


def _default_secret_keywords() -> List[str]:
    return [
        "api key",
        "api keys",
        "access token",
        "refresh token",
        "password",
        "passphrase",
        "private key",
        "ssh key",
        "secret key",
    ]


def _default_insecure_output_patterns() -> List[str]:
    return [
        "store passwords in plain text",
        "store password in plain text",
        "disable authentication",
        "bypass authentication",
        "bypass the firewall",
        "turn off encryption",
        "no need to hash passwords",
    ]


def _analyze_input_security(text_full: str, policy: ACFTSecurityPolicy) -> Dict[str, Any]:
    """
    Analyze the input (prompt + reasoning) for:
    - prompt injection / jailbreak attempts
    - secrets / password requests
    - forbidden topics
    """
    lower = text_full.lower()

    # Injection / jailbreak patterns
    patterns = _default_injection_patterns() + policy.forbid_patterns
    has_injection_pattern = any(pat in lower for pat in patterns)

    # Jailbreak / dev-mode style
    jailbreak_keywords = [
        "developer mode",
        "dev mode",
        "jailbreak",
        "no safety",
        "disable safety",
    ]
    has_jailbreak_pattern = any(k in lower for k in jailbreak_keywords)

    # Secrets / passwords requests
    secret_keywords = _default_secret_keywords()
    secrets_request = any(k in lower for k in secret_keywords)

    # Forbidden topics
    topic_violations: List[str] = []
    for topic in policy.forbid_topics:
        if topic.lower() in lower:
            topic_violations.append(topic)

    return {
        "has_injection_pattern": has_injection_pattern,
        "has_jailbreak_pattern": has_jailbreak_pattern,
        "secrets_request": secrets_request,
        "topic_violations": topic_violations,
    }


def _analyze_output_security(answer: Optional[str], policy: ACFTSecurityPolicy) -> Dict[str, Any]:
    """
    Analyze final answer for:
    - insecure advice
    - secrets leak patterns
    """
    if answer is None:
        return {
            "insecure_advice": False,
            "leak_risk": False,
        }

    lower = answer.lower()

    # Insecure advice patterns (built-in + policy-defined)
    patterns = _default_insecure_output_patterns() + policy.scan_output_for
    insecure_advice = any(pat in lower for pat in patterns)

    # Secret leaking hints (same keywords as secrets request)
    secret_keywords = _default_secret_keywords()
    leak_risk = any(k in lower for k in secret_keywords)

    return {
        "insecure_advice": insecure_advice,
        "leak_risk": leak_risk,
    }


def analyze_security(
    prompt: str,
    reasoning_steps: List[str],
    answer: Optional[str],
    policy: Optional[ACFTSecurityPolicy],
    security_mode_enabled: bool,
) -> SecurityAnalysisResult:
    """
    High-level security analysis used by ACFTEngine.

    If security_mode_enabled is False or policy is None:
      -> returns a SecurityAnalysisResult with risk_level="UNKNOWN".
    Otherwise:
      -> combines input & output scans and computes a risk_level.
    """
    if not security_mode_enabled or policy is None:
        return SecurityAnalysisResult(
            risk_level="UNKNOWN",
            policy_violations=[],
            input_flags={},
            output_flags={},
            notes=[],
            policy_label=None,
        )

    # Combine prompt + reasoning as "input text"
    text_full = " ".join([prompt] + reasoning_steps)

    input_flags = _analyze_input_security(text_full, policy)
    output_flags = _analyze_output_security(answer, policy)

    notes: List[str] = []
    policy_violations: List[str] = list(input_flags["topic_violations"])

    # Base risk
    risk = "LOW"

    # Escalate based on injection / jailbreak / secrets
    if input_flags["has_injection_pattern"] or input_flags["has_jailbreak_pattern"]:
        risk = "HIGH"
        notes.append("Detected prompt injection / jailbreak pattern in input.")

    if input_flags["secrets_request"]:
        risk = "HIGH"
        notes.append("Detected request for secrets / passwords / tokens.")

    if policy_violations and risk != "HIGH":
        risk = "MEDIUM"
        notes.append(
            "Detected forbidden topics: " + ", ".join(policy_violations)
        )

    # Escalate based on answer
    if output_flags["insecure_advice"]:
        risk = "HIGH"
        notes.append("Detected insecure security advice in model answer.")

    if output_flags["leak_risk"]:
        risk = "HIGH"
        notes.append("Detected potential secret-leak pattern in model answer.")

    if not notes:
        notes.append("No obvious security issues detected.")

    return SecurityAnalysisResult(
        risk_level=risk,
        policy_violations=policy_violations,
        input_flags=input_flags,
        output_flags=output_flags,
        notes=notes,
        policy_label=policy.label,
    )


# ---------------------------------------------------------
# JSON-based policy loading from settings / env
# ---------------------------------------------------------

def load_security_policy_from_settings(settings_obj) -> Optional[ACFTSecurityPolicy]:
    """
    Given ACFTSettings, try to load a security policy from JSON.

    Uses:
      - settings_obj.security_mode
      - settings_obj.security_policy_file_enable
      - settings_obj.security_policy_filename
      - env ACFT_SECURITY_POLICY_FILE (optional override)

    Returns:
      ACFTSecurityPolicy instance or None if:
        - security_mode is False, or
        - security_policy_file_enable is False, or
        - file not found / invalid.
    """
    if not getattr(settings_obj, "security_mode", False):
        return None

    if not getattr(settings_obj, "security_policy_file_enable", False):
        return None

    # Optional env override:
    env_path = os.getenv("ACFT_SECURITY_POLICY_FILE")
    if env_path:
        policy_path = Path(env_path).expanduser()
    else:
        # Default: project root / settings_obj.security_policy_filename
        # Assume settings module is under acft/config/settings.py, so project root is 3 parents up from this file.
        project_root = Path(__file__).resolve().parents[2]
        filename = getattr(settings_obj, "security_policy_filename", "security_policy.json")
        policy_path = project_root / filename

    if not policy_path.exists():
        # Silent fail -> caller will fallback to built-in policy if needed
        return None

    try:
        with policy_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    # Build ACFTSecurityPolicy from JSON
    return ACFTSecurityPolicy(
        forbid_topics=data.get("forbid_topics", []),
        forbid_patterns=data.get("forbid_patterns", []),
        scan_output_for=data.get("scan_output_for", []),
        label=data.get("label", "json_policy"),
    )