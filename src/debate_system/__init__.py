"""Multi-agent debate system: propose, critique, vote, aggregate."""

from debate_system.config import ConfigError, DebateConfig
from debate_system.llm.base import BackendError, LLMBackend, LLMRequest
from debate_system.models import DebateResult

__version__ = "0.1.0"

__all__ = [
    "BackendError",
    "ConfigError",
    "DebateConfig",
    "DebateResult",
    "LLMBackend",
    "LLMRequest",
    "__version__",
]
