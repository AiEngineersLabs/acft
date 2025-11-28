from __future__ import annotations

from typing import Any, Dict, List, Optional

from .policy import ACFTSecurityPolicy


# ===== Default patterns =====


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


# ===== Low-level helpers =====


def _analyze_input_security(
    text_full: str,
    policy: ACFTSecurityPolicy,
) -> Dict[str, Any]:
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


def _analyze_output_security(
    answer: Optional[str],
    policy: ACFTSecurityPolicy,
) -> Dict[str, Any]:
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


# ===== High-level API used by engine =====


def analyze_security(
    prompt: str,
    reasoning_steps: List[str],
    answer: Optional[str],
    policy: Optional[ACFTSecurityPolicy],
    security_mode: bool,
) -> Dict[str, Any]:
    """
    High-level security analysis used by ACFTEngine:

    - If security_mode is False or policy is None:
        -> return a neutral 'UNKNOWN' result.
    - Otherwise:
        * input_flags: injection, jailbreak, secrets, topics
        * output_flags: insecure advice, leak risk
        * risk_level: LOW / MEDIUM / HIGH
    """

    # Security disabled or no policy configured
    if not security_mode or policy is None:
        return {
            "risk_level": "UNKNOWN",
            "policy_violations": [],
            "input_flags": {},
            "output_flags": {},
            "notes": [],
            "policy_label": None,
        }

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

    return {
        "risk_level": risk,
        "policy_violations": policy_violations,
        "input_flags": input_flags,
        "output_flags": output_flags,
        "notes": notes,
        "policy_label": policy.label,
    }