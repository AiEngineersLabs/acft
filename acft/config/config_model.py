# acft/config/config_model.py
from __future__ import annotations

from pydantic import BaseModel


class ACFTConfigModel(BaseModel):
    """
    Optional typed snapshot of the resolved ACFT configuration.
    Useful if you want to serialize / log the settings in a structured way
    (separate from ACFTSettings which is a BaseSettings).

    This is not strictly required by the core engine, but it's handy for
    debugging or future UIs.
    """

    llm_backend: str
    model_name: str
    base_url: str

    embed_backend: str
    embed_model: str
    embed_base_url: str

    emit_min_stability: float
    regen_min_stability: float
    retrieve_min_stability: float

    use_retrieval: bool
    rag_folder: str

    security_mode: bool
    security_policy_file_enable: bool
    security_policy_filename: str
    security_policy_file: str | None = None

    pde_enabled: bool
    pde_diffusion: float
    pde_dt: float
    pde_steps: int
    topology_enabled: bool

    class Config:
        arbitrary_types_allowed = True