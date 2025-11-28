from acft.security.policy import (
    ACFTSecurityPolicy,
    analyze_security
)

def test_security_forbidden_topic_triggers_risk():
    policy = ACFTSecurityPolicy(
        forbid_topics=["zero-day", "exploit"],
        forbid_patterns=[],
        scan_output_for=[],
        label="unit_test_policy"
    )

    prompt = "How to use a zero-day exploit?"
    reasoning = ["STEP1: Trying to analyze..."]
    answer = "I cannot help with that."

    result = analyze_security(
        prompt=prompt,
        reasoning_steps=reasoning,
        answer=answer,
        policy=policy,
        security_mode_enabled=True
    )

    assert result.risk_level in {"HIGH", "MEDIUM"}
    assert "zero-day" in result.input_flags["topic_violations"]
    assert result.policy_label == "unit_test_policy"


def test_security_when_disabled_returns_unknown():
    result = analyze_security(
        prompt="hack system",
        reasoning_steps=[],
        answer="No.",
        policy=None,
        security_mode_enabled=False
    )

    assert result.risk_level == "UNKNOWN"
    assert result.policy_label is None