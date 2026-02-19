# Contributing

Thank you for your interest in contributing! This project welcomes improvements to the educational prototype.

> ⚠️ **Reminder**: This is an educational tool. Please do not contribute features intended for clinical use.

## Getting Started

1. **Fork** the repository on GitHub and **clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/diabetes-glucose-pattern-explainer.git
   cd diabetes-glucose-pattern-explainer
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-improvement
   ```

3. **Install dev dependencies** (use a virtual environment):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Development Workflow

### Run tests
```bash
pytest
```

### Run linter
```bash
ruff check src tests
```

### Auto-fix lint issues
```bash
ruff check --fix src tests
```

## Pull Request Guidelines

- Keep PRs focused and small — one logical change per PR.
- Ensure all tests pass and lint is clean before opening a PR.
- Add or update tests for any new functionality.
- Update documentation (docstrings, README, MODEL_CARD.md) as needed.
- Do not introduce real patient data or clinically-oriented features.

## Reporting Issues

Open a GitHub Issue describing the problem, steps to reproduce, and your environment (Python version, OS, package versions).
