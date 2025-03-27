# Contributing to GIRAFFE

GIRAFFE is an open-source project, and contributions are welcome. This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/damiankucharski/GIRAFFE.git
   cd GIRAFFE
   ```

2. Install uv if you don't have it already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Sync the environment with all dependencies:
   ```bash
   uv sync --all-groups --all-extras
   ```

   This will automatically:
   - Create a virtual environment if needed
   - Update the lock file if necessary
   - Install the project in editable mode
   - Install all dependencies including development dependencies and optional extras

4. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

GIRAFFE uses:

- [Ruff](https://github.com/charliermarsh/ruff) for code linting and formatting
- [mypy](https://mypy.readthedocs.io/) for type checking

Docstrings should follow the Google-style format, which is used throughout the codebase.

## Testing

Run the tests using pytest:

```bash
pytest
```

When adding new features, please include tests.

## Documentation

Documentation is written in Markdown and built using MkDocs with the Material theme. API documentation is automatically generated from docstrings using mkdocstrings.

To preview the documentation locally:

```bash
mkdocs serve
```

Then visit `http://127.0.0.1:8000` in your browser.

## Pull Request Process

1. Fork the repository
2. Create a feature branch for your changes
3. Make your changes
4. Run the tests and make sure they pass
5. Update documentation as needed
6. Submit a pull request

## Reporting Issues

If you find a bug or have a feature request, please create an issue in the GitHub repository. Please include:

- A clear and descriptive title
- A description of the issue or feature request
- Steps to reproduce the issue (for bugs)
- Any relevant code samples, error messages, or screenshots
