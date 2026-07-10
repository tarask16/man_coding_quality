"""Тесты модели генерации исходного сообщения M."""

import pytest

from manual_coding_sim.message_model import (
    MessageGenerationConfig,
    MessageModel,
    messages_to_rows,
    summarize_message,
)


def test_message_model_generates_message_with_valid_structure() -> None:
    """Модель должна формировать сообщение M с корректными элементами."""
    config = MessageGenerationConfig(min_length=4, max_length=4)
    model = MessageModel(config=config, random_seed=42)

    message = model.generate_message(message_id="M_TEST")

    assert message.message_id == "M_TEST"
    assert len(message.elements) == 4
    assert message.metadata["message_class_id"] == "G_001"
    assert message.metadata["message_length"] == 4

    for expected_position, element in enumerate(message.elements):
        assert element.position == expected_position
        assert element.element_type in config.element_types
        assert 0.0 <= element.criticality <= 1.0
        assert element.value


def test_message_model_is_reproducible_with_same_seed() -> None:
    """Одинаковый random_seed должен давать одинаковое сообщение M."""
    config = MessageGenerationConfig(min_length=6, max_length=6)
    first_model = MessageModel(config=config, random_seed=123)
    second_model = MessageModel(config=config, random_seed=123)

    first_message = first_model.generate_message()
    second_message = second_model.generate_message()

    assert first_message == second_message


def test_message_model_reset_restores_initial_sequence() -> None:
    """Сброс модели должен восстанавливать начальную последовательность M."""
    model = MessageModel(random_seed=777)

    first_message = model.generate_message()
    model.generate_message()
    model.reset()
    repeated_first_message = model.generate_message()

    assert first_message == repeated_first_message


def test_generate_batch_creates_unique_message_ids() -> None:
    """Пакет сообщений должен иметь стабильные уникальные идентификаторы."""
    model = MessageModel(random_seed=42)

    messages = model.generate_batch(3)
    message_ids = [message.message_id for message in messages]

    assert message_ids == ["M_000001", "M_000002", "M_000003"]


def test_generate_batch_rejects_non_positive_count() -> None:
    """Число сообщений в пакете должно быть положительным."""
    model = MessageModel(random_seed=42)

    with pytest.raises(ValueError, match="положительным"):
        model.generate_batch(0)


def test_message_summary_contains_control_fields() -> None:
    """Сводка сообщения должна содержать контрольные поля."""
    config = MessageGenerationConfig(min_length=5, max_length=5)
    model = MessageModel(config=config, random_seed=42)
    message = model.generate_message(message_id="M_SUMMARY")

    summary = summarize_message(message)
    rows = messages_to_rows((message,))

    assert summary["message_id"] == "M_SUMMARY"
    assert summary["message_length"] == 5
    assert 0.0 <= summary["mean_criticality"] <= 1.0
    assert rows == [summary]


def test_message_generation_config_validation() -> None:
    """Конфигурация класса сообщений G должна проверять некорректные значения."""
    invalid_config = MessageGenerationConfig(min_length=10, max_length=5)

    with pytest.raises(ValueError, match="Максимальная длина"):
        MessageModel(config=invalid_config, random_seed=42)
