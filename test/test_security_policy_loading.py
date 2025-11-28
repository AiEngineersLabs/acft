from pathlib import Path
from acft.config import ACFTSettings
from acft.security.policy import (
    load_security_policy_from_settings,
    ACFTSecurityPolicy,
)

def test_load_policy_disabled_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    (tmp_path / "security_policy.json").write_text("{}")

    settings = ACFTSettings(
        security_mode=True,
        security_policy_file_enable=False,
        security_policy_filename="security_policy.json"
    )

    policy = load_security_policy_from_settings(settings)
    assert policy is None