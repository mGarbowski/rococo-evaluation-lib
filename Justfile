# Install dependencies
install:
    uv sync

# Run unit tests
test:
    uv run pytest

# Install this package into the environment
install_lib:
    uv pip install -e .

# Clean build artifacts
clean:
    rm -rf .pytest_cache .coverage dist build **/__pycache__ *.egg-info src/*.egg-info
