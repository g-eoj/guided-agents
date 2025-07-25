"""Tests for utility functions in models.py"""

from dataclasses import dataclass
from typing import Optional

from guided_agents.models import (
    ChatMessageToolCall,
    ChatMessageToolCallDefinition,
    MessageRole,
    get_dict_from_nested_dataclasses,
    parse_json_if_needed,
    remove_stop_sequences,
)


@dataclass
class SampleDataClass:
    name: str
    value: int
    nested: Optional["SampleDataClass"] = None


@dataclass
class SampleNestedDataClass:
    outer: SampleDataClass
    description: str


class TestUtilityFunctions:
    """Test utility functions from models.py"""

    def test_get_dict_from_nested_dataclasses_simple(self):
        """Test converting simple dataclass to dict"""
        obj = SampleDataClass(name="test", value=42)
        result = get_dict_from_nested_dataclasses(obj)
        expected = {"name": "test", "value": 42, "nested": None}
        assert result == expected

    def test_get_dict_from_nested_dataclasses_with_nesting(self):
        """Test converting nested dataclass to dict"""
        inner = SampleDataClass(name="inner", value=1)
        outer = SampleDataClass(name="outer", value=2, nested=inner)
        result = get_dict_from_nested_dataclasses(outer)
        expected = {
            "name": "outer",
            "value": 2,
            "nested": {"name": "inner", "value": 1, "nested": None}
        }
        assert result == expected

    def test_get_dict_from_nested_dataclasses_with_ignore_key(self):
        """Test ignoring specified key during conversion"""
        obj = SampleDataClass(name="test", value=42)
        result = get_dict_from_nested_dataclasses(obj, ignore_key="value")
        expected = {"name": "test", "nested": None}
        assert result == expected

    def test_get_dict_from_nested_dataclasses_non_dataclass(self):
        """Test handling non-dataclass objects"""
        obj = {"not": "a_dataclass"}
        result = get_dict_from_nested_dataclasses(obj)
        assert result == obj

    def test_parse_json_if_needed_with_dict(self):
        """Test parse_json_if_needed with dict input"""
        input_dict = {"key": "value", "number": 42}
        result = parse_json_if_needed(input_dict)
        assert result == input_dict

    def test_parse_json_if_needed_with_valid_json_string(self):
        """Test parse_json_if_needed with valid JSON string"""
        json_string = '{"key": "value", "number": 42}'
        result = parse_json_if_needed(json_string)
        expected = {"key": "value", "number": 42}
        assert result == expected

    def test_parse_json_if_needed_with_invalid_json_string(self):
        """Test parse_json_if_needed with invalid JSON string"""
        invalid_json = "not a json string"
        result = parse_json_if_needed(invalid_json)
        assert result == invalid_json

    def test_parse_json_if_needed_with_empty_string(self):
        """Test parse_json_if_needed with empty string"""
        result = parse_json_if_needed("")
        assert result == ""

    def test_remove_stop_sequences_single_sequence(self):
        """Test removing single stop sequence from end of content"""
        content = "Hello world<|end|>"
        stop_sequences = ["<|end|>"]
        result = remove_stop_sequences(content, stop_sequences)
        assert result == "Hello world"

    def test_remove_stop_sequences_multiple_sequences(self):
        """Test removing multiple stop sequences"""
        content = "Hello world<|stop|>"
        stop_sequences = ["<|end|>", "<|stop|>"]
        result = remove_stop_sequences(content, stop_sequences)
        assert result == "Hello world"

    def test_remove_stop_sequences_no_match(self):
        """Test when no stop sequences are found"""
        content = "Hello world"
        stop_sequences = ["<|end|>", "<|stop|>"]
        result = remove_stop_sequences(content, stop_sequences)
        assert result == content

    def test_remove_stop_sequences_partial_match(self):
        """Test when stop sequence appears in middle, not end"""
        content = "Hello <|end|> world"
        stop_sequences = ["<|end|>"]
        result = remove_stop_sequences(content, stop_sequences)
        assert result == content  # Should not remove from middle

    def test_remove_stop_sequences_empty_content(self):
        """Test with empty content"""
        content = ""
        stop_sequences = ["<|end|>"]
        result = remove_stop_sequences(content, stop_sequences)
        assert result == ""

    def test_remove_stop_sequences_empty_sequences(self):
        """Test with empty stop sequences list"""
        content = "Hello world"
        stop_sequences = []
        result = remove_stop_sequences(content, stop_sequences)
        assert result == content


class TestDataClasses:
    """Test dataclass structures from models.py"""

    def test_chat_message_tool_call_definition_creation(self):
        """Test creating ChatMessageToolCallDefinition"""
        definition = ChatMessageToolCallDefinition(
            arguments={"param": "value"},
            name="test_function",
            description="A test function"
        )
        assert definition.arguments == {"param": "value"}
        assert definition.name == "test_function"
        assert definition.description == "A test function"

    def test_chat_message_tool_call_definition_optional_description(self):
        """Test ChatMessageToolCallDefinition with optional description"""
        definition = ChatMessageToolCallDefinition(
            arguments={"param": "value"},
            name="test_function"
        )
        assert definition.description is None

    def test_chat_message_tool_call_creation(self):
        """Test creating ChatMessageToolCall"""
        definition = ChatMessageToolCallDefinition(
            arguments={"param": "value"},
            name="test_function"
        )
        tool_call = ChatMessageToolCall(
            function=definition,
            id="call_123",
            type="function"
        )
        assert tool_call.function == definition
        assert tool_call.id == "call_123"
        assert tool_call.type == "function"

    def test_message_role_enum(self):
        """Test MessageRole enum values"""
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.TOOL_CALL == "tool-call"
        assert MessageRole.TOOL_RESPONSE == "tool-response"

    def test_message_role_roles_method(self):
        """Test MessageRole.roles() method"""
        roles = MessageRole.roles()
        expected = ["user", "assistant", "system", "tool-call", "tool-response"]
        assert set(roles) == set(expected)
        assert len(roles) == len(expected)
