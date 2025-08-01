### 🔄 Project Awareness & Context

- **This CLAUDE.md file serves as the central planning document** containing the project's architecture, goals, style, and constraints.
- **Use GitHub Issues/Projects for task management** - Check existing issues before starting a new task. Create new issues for tasks that aren't tracked.
- **For VA pipeline tasks**, check the appropriate documentation base on the task, either `baseline_benchmark.md`, `transfer_learning.md`, or `active_learning.md`.
- **Use consistent naming conventions, file structure, and architecture patterns** as described in this document.
- **Use venv_linux** (the virtual environment) whenever executing Python commands, including for unit tests.
- ****CRITICAL:** **Poetry is used for dependency management** - use `poetry install` and `poetry add` for package management.
- ****CRITICAL:** **Do NOT use mock model/lib/data expect testing**

## Core Principles

**IMPORTANT: You MUST follow these principles in all code changes and PRP generations:**

### KISS (Keep It Simple, Stupid)

- Simplicity should be a key goal in design
- Choose straightforward solutions over complex ones whenever possible
- Simple solutions are easier to understand, maintain, and debug

### YAGNI (You Aren't Gonna Need It)

- Avoid building functionality on speculation
- Implement features only when they are needed, not when you anticipate they might be useful in the future

### Open/Closed Principle

- Software entities should be open for extension but closed for modification
- Design systems so that new functionality can be added with minimal changes to existing code


### Essential poetry Commands

poetry run

### 🧱 Code Structure & Modularity

- **Never create a file longer than 350 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
  For the VA pipeline:
  - `baseline/` - Baseline benchmark implementation
  - `transfer/` - Transfer learning components
  - `active/` - Active learning modules
  - `models/` - Model implementations
  - `data/` - Data processing utilities
- **Use clear, consistent imports** (prefer relative imports within packages).
- **Use python_dotenv and load_env()** for environment variables.

### 🧪 Testing & Reliability

- **Always create Pytest unit tests for new features** (functions, classes, routes, etc).
- **After updating any logic**, check whether existing unit tests need to be updated. If so, do it.
- **Tests should live in a `/tests` folder** mirroring the main app structure.
  - Include at least:
    - 1 test for expected use
    - 1 edge case
    - 1 failure case
- **For VA models**, include tests for:
  - Data preprocessing pipeline
  - Model training/prediction
  - Metric calculations (CSMF accuracy, COD accuracy)
  - Stratification logic

### ✅ Task Completion

- **Update GitHub Issues with progress** - Add brief comments about approach and any blockers encountered during development.
- **Link PRs to Issues** - Use keywords like `Fixes #123` in PR descriptions to auto-close issues when merged.
- **Create new GitHub Issues** for any sub-tasks or TODOs discovered during development, linking them to the parent issue when applicable.
- **Follow team's issue closing policy** - Issues typically close on PR merge, not immediately after code completion.
- **For pipeline deliverables**, ensure output files are saved in appropriate directories:
  - `results/baseline/benchmark_results.csv`
  - `results/transfer/transfer_results.csv`
  - `results/active/active_learning_results.csv`
- **After finish implementation, always run it once as confirmation. If it is timeout, you can assume it pass and tell the user run it itself**

### 🔄 Development Workflow

- **Branch Naming Conventions**:

  - Feature branches: `feature/issue-123-brief-description`
  - Bug fixes: `fix/issue-123-brief-description`
  - Hotfixes: `hotfix/critical-issue-description`
  - Always include issue number when applicable
- **Commit Message Standards**:

  - Follow conventional commits: `type(scope): description`
  - Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
  - Example: `feat(auth): add JWT token validation`
  - Keep first line under 72 characters
  - Add detailed description after blank line if needed

### 📎 Style & Conventions

- **Use Python** as the primary language.
- **Follow PEP8**, use type hints, and format with `black`.
- **Use `pydantic` for data validation**.
- **Use `pandas` for data manipulation** and `scikit-learn` for ML utilities.
- **For VA-specific algorithms (openVA, InSilicoVA, InterVA)**:
  - Use the Docker image provided at `models/insilico/Dockerfile`
  - Keep R code isolated within Docker containers
  - Use Python to orchestrate Docker container calls
  - Document any new R dependencies in the Dockerfile
- Write **docstrings for every function** using the Google style:
  ```python
  def example():
      """
      Brief summary.

      Args:
          param1 (type): Description.

      Returns:
          type: Description.
      """
  ```

### 📚 Documentation & Explainability

- **Update `README.md`** when new features are added, dependencies change, or setup steps are modified.
- **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.

### 🧠 AI Behavior Rules

- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use known, verified Python packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a documented GitHub Issue.
- **For VA-specific terms**, use standard terminology (COD, CSMF, VA, etc.) consistently.
- **Use context7 MCP for library documentation** - When you need current documentation for libraries (scikit-learn, pandas, numpy, openVA, etc.), use the context7 MCP tools instead of relying on potentially outdated knowledge.

### ⏱️ Execution Time Constraints

- **Claude Code has a 5-minute execution timeout** for any single command.
- **For long-running computations** (e.g., extensive model training, large-scale cross-validation):

  - Create standalone Python scripts that users can run manually
  - Make scripts executable with proper shebang (`#!/usr/bin/env python`)
  - Include clear usage instructions at the top of the script:
    ```python
    """
    Long-running VA model training script

    Usage: python train_models.py --data path/to/data.csv

    Expected runtime: ~2 hours for full cross-validation
    Progress will be saved to checkpoints/ directory
    """
    ```
  - Implement checkpointing to allow resuming interrupted runs
  - Add progress indicators using `tqdm` or logging
  - Log intermediate results for debugging
- **Design considerations for manual execution scripts**:

  - Use argparse for command-line arguments
  - Provide sensible defaults
  - Include `--dry-run` option for testing
  - Save outputs incrementally, not just at the end
  - Add verbose logging with timestamps

### 🔒 Data Privacy & Security

- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use known, verified Python packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a task