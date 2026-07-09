        """
        Проверки базовой структуры пакета исследовательского симулятора.

        Тесты подтверждают, что каркас пакета импортируется, конфигурация
        эксперимента загружается, а базовые типы данных создаются без ошибок.
        """

        from pathlib import Path

        from manual_coding_sim import (
            FeatureGroup,
            GeneratedMessage,
            MessageElement,
            QualityVector,
            ScenarioParameters,
            load_experiment_config,
        )


        def test_load_experiment_config(tmp_path: Path) -> None:
            """Проверить загрузку базовой конфигурации эксперимента."""

            config_path = tmp_path / "base_experiment.yaml"
            config_path.write_text(
                "experiment:
"
                "  name: chapter3_base_environment_check
"
                "  random_seed: 42
",
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

            assert config.name == "chapter3_base_environment_check"
            assert config.random_seed == 42


        def test_scenario_parameters_describe_full_scenario() -> None:
            """Проверить структуру сценария A = {S, O, U, G, K}."""

            scenario = ScenarioParameters(
                scenario_id="scenario_001",
                coding_tool_id="S_basic",
                operator_id="O_regular",
                condition_id="U_normal",
                message_class_id="G_short",
                control_procedure_id="K_double_check",
            )

            assert scenario.scenario_id == "scenario_001"
            assert scenario.coding_tool_id.startswith("S_")
            assert scenario.operator_id.startswith("O_")
            assert scenario.condition_id.startswith("U_")
            assert scenario.message_class_id.startswith("G_")
            assert scenario.control_procedure_id.startswith("K_")


        def test_generated_message_contains_elements() -> None:
            """Проверить представление исходного сообщения M."""

            element = MessageElement(
                value="A1",
                element_type="symbol",
                position=0,
                criticality=0.75,
            )
            message = GeneratedMessage(
                message_id="M_001",
                message_class_id="G_short",
                elements=(element,),
            )

            assert message.elements[0].value == "A1"
            assert message.elements[0].criticality == 0.75


        def test_quality_vector_contains_partial_quality_indicators() -> None:
            """Проверить вектор частных показателей качества."""

            quality = QualityVector(
                q_acc=0.95,
                q_time=0.80,
                q_effort=0.70,
                q_res=0.90,
                q_rep=0.85,
                q_fit=0.88,
            )

            assert quality.q_acc > quality.q_effort
            assert FeatureGroup.PRIOR == "prior"
