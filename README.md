# Guided Agents

This is a fork of the awesome [smolagents](https://github.com/huggingface/smolagents) library.
The goal of this fork is to integrate structured output.

## Why structured output?

### Reliable outputs

Instead of hoping one or few shot prompting will produce the desired output, structured output allows you to define a schema (or pretty much any structure) for the output.
This way, you can ensure that the output is always in a parsable format without prompt engineering time waste.

### Reduced token consumption

#### Control reasoning

Structured output opens up different strategies for limiting reasoning output.
This can make a difference with token hungry reasoning models while still maintaining good performance.

#### Short system prompts

Another advantage of not requiring one/few shot prompting is faster processing and reduced token consumption.
It is worth noting prompt caching has a similar effect.


## Where to start?

The is repo is a playground, and thus breaking changes are expected.
That said, please look at the [examples](https://github.com/g-eoj/guided-agents/blob/main/example/core.py) to get an idea of how these ideas come together.

## Development

### Testing

This project uses pytest for testing. To run the tests:

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage
pytest --cov=guided_agents

# Run specific test file
pytest tests/test_models.py -v
```

### Linting

We use ruff for code formatting and linting:

```bash
# Install ruff
pip install ruff

# Check for issues
ruff check src/ tests/

# Format code
ruff format src/ tests/
```
