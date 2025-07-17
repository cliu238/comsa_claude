### 🔄 Project Awareness & Context

- **This CLAUDE.md file serves as the central planning document** containing the project's architecture, goals, style, and constraints.
- **Use GitHub Issues/Projects for task management** - Check existing issues before starting a new task. Create new issues for tasks that aren't tracked.
- **For VA pipeline tasks**, check the appropriate documentation base on the task, either `baseline_benchmark.md`, `transfer_learning.md`, or `active_learning.md`.
- **Use consistent naming conventions, file structure, and architecture patterns** as described in this document.
- **Use venv_linux** (the virtual environment) whenever executing Python commands, including for unit tests.
- **Poetry is used for dependency management** - use `poetry install` and `poetry add` for package management.

### 🧱 Code Structure & Modularity

- **Never create a file longer than 350 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
  For the VA pipeline:
  - `baseline/` - Baseline benchmark implementation
  - `transfer/` - Transfer learning components
  - `active/` - Active learning modules
  - `models/` - Model implementations
  - `data/` - Data processing utilities
  - `evaluation/` - Metrics and evaluation functions
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

### 🔄 Development Workflow

- **Branch Naming Conventions**:
  - Feature branches: `feature/issue-123-brief-description`
  - Bug fixes: `fix/issue-123-brief-description`
  - Hotfixes: `hotfix/critical-issue-description`
  - Always include issue number when applicable

- **Pull Request Guidelines**:
  - **PR Title**: Clear, descriptive summary (e.g., "Add user authentication with JWT")
  - **PR Description Template**:
    ```
    ## Summary
    Brief description of changes
    
    ## Related Issue
    Fixes #123 (or Closes #123, Resolves #123)
    
    ## Changes Made
    - List key changes
    - Include any breaking changes
    
    ## Testing
    - How to test the changes
    - Any specific test cases covered
    ```
  - **Draft PRs**: Use for work-in-progress to get early feedback
  - **Reviews**: Wait for at least one approval before merging

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
- **For VA-specific algorithms**, use the OpenVA library via Docker when needed.
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

### 📊 VA Pipeline Specific Guidelines

- **Data Handling**:

  - Always validate VA data format before processing
  - Use standardized column names across all datasets
  - Implement age-group and site stratification for splits
  - Handle missing data appropriately for each algorithm
- **Model Training**:

  - Use 5-fold cross-validation for hyperparameter tuning
  - Implement stratified train/test splits
  - Save trained models with versioning
  - Log all hyperparameters and results
- **Evaluation**:

  - Always calculate CSMF accuracy, Top-1/Top-3 COD accuracy
  - Generate confusion matrices for cause assignments
  - Create comparison tables across all models
  - Export results in standardized CSV format
- **Transfer Learning**:

  - Document source and target domain mappings
  - Implement domain adaptation techniques from ADAPT library
  - Track performance degradation/improvement
- **Active Learning**:

  - Implement uncertainty sampling strategies
  - Track annotation budget and efficiency
  - Document query selection criteria

### 📚 Documentation & Explainability

- **Update `README.md`** when new features are added, dependencies change, or setup steps are modified.
- **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.
- **For each pipeline component**, maintain separate documentation:
  - `baseline_benchmark.md` - Baseline methodology and results
  - `transfer_learning.md` - Transfer learning approach
  - `active_learning.md` - Active learning strategy

### 🧠 AI Behavior Rules

- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use known, verified Python packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a documented GitHub Issue.
- **For VA-specific terms**, use standard terminology (COD, CSMF, VA, etc.) consistently.

### 🔒 Data Privacy & Security

- **Never commit raw VA data** containing personal identifiers
- **Use anonymized IDs** for all data processing
- **Store sensitive data** only in designated secure directories
- **Document data access requirements** in README
