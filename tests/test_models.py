"""Tests for model classes in models.py"""

from unittest.mock import patch

from guided_agents.models import (
    MessageRole,
    MLXVLMGenerationResult,
    Model,
)


class TestModel:
    """Test the base Model class"""

    def test_model_initialization(self):
        """Test Model initialization with kwargs"""
        model = Model(temperature=0.7, max_tokens=100)
        assert model.last_input_token_count is None
        assert model.last_output_token_count is None
        assert model.kwargs == {"temperature": 0.7, "max_tokens": 100}

    def test_model_initialization_empty(self):
        """Test Model initialization without kwargs"""
        model = Model()
        assert model.last_input_token_count is None
        assert model.last_output_token_count is None
        assert model.kwargs == {}

    @patch('guided_agents.models.get_clean_message_list')
    def test_prepare_completion_kwargs_basic(self, mock_get_clean_message_list):
        """Test _prepare_completion_kwargs with basic parameters"""
        # Setup
        model = Model(temperature=0.8)
        messages = [{"role": "user", "content": "Hello"}]
        mock_get_clean_message_list.return_value = messages

        # Execute
        result = model._prepare_completion_kwargs(messages)

        # Verify
        mock_get_clean_message_list.assert_called_once()
        assert result["messages"] == messages
        assert result["temperature"] == 0.8

    @patch('guided_agents.models.get_clean_message_list')
    def test_prepare_completion_kwargs_with_stop_sequences(self, mock_get_clean_message_list):
        """Test _prepare_completion_kwargs with stop sequences"""
        # Setup
        model = Model()
        messages = [{"role": "user", "content": "Hello"}]
        stop_sequences = ["<|end|>", "<|stop|>"]
        mock_get_clean_message_list.return_value = messages

        # Execute
        result = model._prepare_completion_kwargs(messages, stop_sequences=stop_sequences)

        # Verify
        assert result["stop"] == stop_sequences
        assert result["messages"] == messages

    @patch('guided_agents.models.get_clean_message_list')
    def test_prepare_completion_kwargs_with_guide(self, mock_get_clean_message_list):
        """Test _prepare_completion_kwargs with guide parameter"""
        # Setup
        model = Model()
        messages = [{"role": "user", "content": "Hello"}]
        guide = "some guide string"
        mock_get_clean_message_list.return_value = messages

        # Execute
        result = model._prepare_completion_kwargs(messages, guide=guide)

        # Verify
        assert result["guide"] == guide
        assert result["messages"] == messages

    @patch('guided_agents.models.get_clean_message_list')
    def test_prepare_completion_kwargs_kwarg_override(self, mock_get_clean_message_list):
        """Test that explicit kwargs override model defaults"""
        # Setup
        model = Model(temperature=0.5, max_tokens=50)
        messages = [{"role": "user", "content": "Hello"}]
        mock_get_clean_message_list.return_value = messages

        # Execute - explicit kwargs should override defaults
        result = model._prepare_completion_kwargs(
            messages,
            temperature=0.9,  # Override default
            extra_param="new_value"
        )

        # Verify
        assert result["temperature"] == 0.9  # Overridden
        assert result["max_tokens"] == 50    # From model defaults
        assert result["extra_param"] == "new_value"  # New parameter
        assert result["messages"] == messages


class TestMLXVLMGenerationResult:
    """Test MLXVLMGenerationResult dataclass"""

    def test_creation_with_all_fields(self):
        """Test creating MLXVLMGenerationResult with all fields"""
        result = MLXVLMGenerationResult(
            text="Hello world",
            token=123,
            logprobs=[0.1, 0.2, 0.3],
            prompt_tokens=10,
            generation_tokens=20,
            prompt_tps=100.5,
            generation_tps=50.25,
            peak_memory=1024.0,
            finish_reason="stop"
        )

        assert result.text == "Hello world"
        assert result.token == 123
        assert result.logprobs == [0.1, 0.2, 0.3]
        assert result.prompt_tokens == 10
        assert result.generation_tokens == 20
        assert result.prompt_tps == 100.5
        assert result.generation_tps == 50.25
        assert result.peak_memory == 1024.0
        assert result.finish_reason == "stop"

    def test_creation_with_required_fields_only(self):
        """Test creating MLXVLMGenerationResult with only required fields"""
        result = MLXVLMGenerationResult(
            text="Hello",
            token=None,
            logprobs=None,
            prompt_tokens=5,
            generation_tokens=10,
            prompt_tps=75.0,
            generation_tps=25.0,
            peak_memory=512.0
        )

        assert result.text == "Hello"
        assert result.token is None
        assert result.logprobs is None
        assert result.prompt_tokens == 5
        assert result.generation_tokens == 10
        assert result.prompt_tps == 75.0
        assert result.generation_tps == 25.0
        assert result.peak_memory == 512.0
        assert result.finish_reason is None  # Default value

    def test_creation_with_optional_finish_reason(self):
        """Test creating MLXVLMGenerationResult with optional finish_reason"""
        result = MLXVLMGenerationResult(
            text="Test",
            token=42,
            logprobs=[],
            prompt_tokens=1,
            generation_tokens=1,
            prompt_tps=1.0,
            generation_tps=1.0,
            peak_memory=100.0,
            finish_reason="length"
        )

        assert result.finish_reason == "length"


class TestModelUtilityIntegration:
    """Integration tests for model utility functions"""

    def test_model_with_real_message_structure(self):
        """Test Model class with realistic message structures"""
        model = Model(temperature=0.7)

        messages = [
            {"role": MessageRole.SYSTEM, "content": "You are a helpful assistant."},
            {"role": MessageRole.USER, "content": "Hello, how are you?"},
            {"role": MessageRole.ASSISTANT, "content": "I'm doing well, thank you!"}
        ]

        # This should work without errors when get_clean_message_list is properly mocked
        with patch('guided_agents.models.get_clean_message_list') as mock_clean:
            mock_clean.return_value = messages
            result = model._prepare_completion_kwargs(messages)

            assert "messages" in result
            assert result["temperature"] == 0.7
            mock_clean.assert_called_once()
