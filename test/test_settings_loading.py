from acft.config import load_acft_settings

def test_env_loading(monkeypatch):
    monkeypatch.setenv("ACFT_LLAMA_MODEL", "llama3.2:latest")
    monkeypatch.setenv("ACFT_SECURITY_MODE", "true")

    settings = load_acft_settings()

    assert settings.model_name == "llama3.2:latest"
    assert settings.security_mode is True