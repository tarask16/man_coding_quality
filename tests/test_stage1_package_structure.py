"""Тесты базовой структуры пакета исследовательского симулятора."""

from manual_coding_sim import __version__, load_experiment_config
from manual_coding_sim.types import (
    FeatureGroup,
    GeneratedMessage,
    MessageElement,
    QualityVector,
    ScenarioParameters,
)


def test_package_import() -> None:
    """Пакет должен импортироваться из каталога src."""
    assert __version__ == "0.1.0"


def test_load_experiment_config() -> None:
    """Конфигурация эксперимента должна содержать имя и random_seed."""
    config = load_experiment_config("configs/base_experiment.yaml")

    assert config["experiment"]["name"]
    assert isinstance(config["experiment"]["random_seed"], int)


def test_create_scenario_and_message() -> None:
    """Базовые типы должны описывать сценарий A и сообщение M."""
    scenario = ScenarioParameters(
        scenario_id="A_001",
        coding_tool_id="S_001",
        operator_id="O_001",
        condition_id="U_001",
        message_class_id="G_001",
        control_procedure_id="K_001",
    )
    element = MessageElement(
        value="alpha",
        element_type="symbolic",
        position=0,
        criticality=0.7,
    )
    message = GeneratedMessage(
        message_id="M_001",
        elements=(element,),
        metadata={"scenario_id": scenario.scenario_id},
    )

    assert scenario.scenario_id == "A_001"
    assert message.elements[0].value == "alpha"
    assert message.metadata["scenario_id"] == "A_001"


def test_quality_vector_and_feature_group() -> None:
    """Показатели качества и группы признаков должны создаваться явно."""
    quality = QualityVector(
        q_acc=0.95,
        q_time=0.80,
        q_effort=0.60,
        q_res=0.75,
        q_rep=0.90,
        q_fit=0.85,
    )
    features = FeatureGroup(
        scenario_id="A_001",
        prior_features={"procedure_complexity": 0.5},
        fact_features={"error_rate": 0.05},
        diagnostic_features={"detected_errors": 2.0},
    )

    assert quality.q_acc > 0.9
    assert "procedure_complexity" in features.prior_features
    assert "error_rate" in features.fact_features
