"""Тесты токенизации априорных признаков для LDA-корпуса."""

from manual_coding_sim.lda.config import LdaTokenizationConfig
from manual_coding_sim.lda.tokenization import FeatureTokenizer


ROWS = [
    {
        "protocol_id": "p001",
        "scenario_id": "s001",
        "has_control": "1",
        "message_length": "10",
        "procedure_type": "simple",
        "comment": "",
    },
    {
        "protocol_id": "p002",
        "scenario_id": "s002",
        "has_control": "0",
        "message_length": "20",
        "procedure_type": "complex",
        "comment": "manual check",
    },
    {
        "protocol_id": "p003",
        "scenario_id": "s003",
        "has_control": "1",
        "message_length": "30",
        "procedure_type": "simple",
        "comment": "manual check",
    },
]


def test_feature_tokenizer_ignores_identifier_columns() -> None:
    """Служебные идентификаторы не должны попадать в признаки LDA."""

    tokenizer = FeatureTokenizer().fit(ROWS)

    assert "protocol_id" not in tokenizer.feature_columns
    assert "scenario_id" not in tokenizer.feature_columns
    assert "has_control" in tokenizer.feature_columns


def test_feature_tokenizer_handles_binary_numeric_categorical_and_missing() -> None:
    """Токенизатор должен обрабатывать основные типы априорных признаков."""

    tokenizer = FeatureTokenizer(
        config=LdaTokenizationConfig(numeric_bins=3),
    ).fit(ROWS)

    first_row_tokens = {item.token for item in tokenizer.transform_row(ROWS[0])}
    second_row_tokens = {item.token for item in tokenizer.transform_row(ROWS[1])}
    third_row_tokens = {item.token for item in tokenizer.transform_row(ROWS[2])}

    assert "has_control__present" in first_row_tokens
    assert "has_control__absent" in second_row_tokens
    assert "message_length__level_low" in first_row_tokens
    assert "message_length__level_mid" in second_row_tokens
    assert "message_length__level_high" in third_row_tokens
    assert "procedure_type__value_simple" in first_row_tokens
    assert "procedure_type__value_complex" in second_row_tokens
    assert "comment__missing" in first_row_tokens


def test_feature_tokenizer_token_map_contains_reproducible_rules() -> None:
    """Карта токенизации должна сохранять правила и порядок признаков."""

    tokenizer = FeatureTokenizer(
        config=LdaTokenizationConfig(numeric_strategy="uniform", numeric_bins=3),
    ).fit(ROWS)
    token_map = tokenizer.to_token_map()

    assert token_map["numeric_strategy"] == "uniform"
    assert token_map["numeric_bins"] == 3
    assert "feature_columns" in token_map
    assert token_map["feature_columns"] == [
        "has_control",
        "message_length",
        "procedure_type",
        "comment",
    ]
    assert len(token_map["rules"]) == 4


def test_feature_tokenizer_requires_fit_before_transform() -> None:
    """Преобразование строк до fit должно завершаться ошибкой."""

    tokenizer = FeatureTokenizer()

    try:
        tokenizer.transform_row(ROWS[0])
    except RuntimeError as error:
        assert "fit" in str(error)
    else:
        msg = "Ожидалась ошибка при transform_row до обучения токенизатора."
        raise AssertionError(msg)
