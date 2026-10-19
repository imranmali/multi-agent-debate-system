import pytest

from debate_system.config import DEFAULT_PERSONAS, ConfigError, DebateConfig


def test_defaults_are_valid() -> None:
    config = DebateConfig().validate()
    assert config.num_agents == 3
    assert config.voting_strategy == "weighted"


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"num_agents": 1}, "num_agents"),
        ({"max_rounds": 0}, "max_rounds"),
        ({"consensus_threshold": 0}, "consensus_threshold"),
        ({"consensus_threshold": 1.01}, "consensus_threshold"),
        ({"voting_strategy": "ranked-pairs"}, "voting_strategy"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"personas": ()}, "personas"),
        ({"confidence_weights": {"luck": 1.0}}, "confidence_weights"),
        ({"confidence_weights": {"support": -1.0}}, "confidence_weights"),
        ({"confidence_weights": {"support": 0.0}}, "confidence_weights"),
    ],
)
def test_invalid_config_names_field(kwargs: dict[str, object], field: str) -> None:
    with pytest.raises(ConfigError) as exc:
        DebateConfig(**kwargs).validate()  # type: ignore[arg-type]
    assert exc.value.field == field
    assert field in str(exc.value)


def test_threshold_of_one_is_allowed() -> None:
    DebateConfig(consensus_threshold=1.0).validate()


def test_personas_cycle_and_agent_ids() -> None:
    config = DebateConfig(num_agents=3, personas=("a", "b"))
    assert [config.persona_for(i) for i in range(3)] == ["a", "b", "a"]
    assert config.agent_ids() == ["agent-1", "agent-2", "agent-3"]


def test_to_dict_is_json_friendly() -> None:
    data = DebateConfig().to_dict()
    assert data["personas"] == list(DEFAULT_PERSONAS)
    assert data["confidence_weights"]["support"] == 0.4
