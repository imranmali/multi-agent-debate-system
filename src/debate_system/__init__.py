"""Multi-agent debate system: propose, critique, vote, aggregate."""

from debate_system.config import ConfigError, DebateConfig
from debate_system.llm import BackendError, LLMBackend, LLMRequest, create_backend
from debate_system.models import DebateResult
from debate_system.orchestrator import DebateError, DebateOrchestrator

__version__ = "0.1.0"

__all__ = [
    "BackendError",
    "ConfigError",
    "DebateConfig",
    "DebateError",
    "DebateOrchestrator",
    "DebateResult",
    "LLMBackend",
    "LLMRequest",
    "__version__",
    "create_backend",
]
