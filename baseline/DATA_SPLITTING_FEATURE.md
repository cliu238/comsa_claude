## FEATURE:

- Data splitting module for the baseline VA pipeline that segments datasets by site for future analysis
- Site-based stratification functionality to ensure proper data distribution across different geographic locations
- Test/training split functionality with configurable ratios and stratified sampling
- Imbalanced class handling for small label classes with insufficient sample sizes
- Cross-validation splitting strategies for model evaluation and validation
- Configurable splitting parameters with validation to handle various site configurations
- Export functionality for split datasets with proper metadata tracking

## EXAMPLES:

In the `baseline/` folder, there should be examples of:

- `baseline/data_splitter.py` - main module for site-based and train/test data splitting with clean, simple logic
- `baseline/utils/site_validator.py` - utility functions for validating site configurations
- `baseline/utils/class_validator.py` - utility functions for handling imbalanced classes and minimum sample validation
- `baseline/stratified_splitter.py` - stratified sampling implementation for maintaining class distribution
- `tests/test_data_splitter.py` - comprehensive unit tests for the splitting functionality
- `tests/test_imbalanced_handling.py` - tests for small class handling and edge cases

Examples should demonstrate:
- Train/test splitting with configurable ratios (e.g., 80/20, 70/30)
- Stratified splitting that preserves class distribution
- Handling of classes with fewer than minimum required samples
- Cross-validation fold generation for model evaluation
- Site-based splitting combined with train/test splits

The current implementation may be over-complicated and should be simplified following KISS principles.

**IMPORTANT: Keep logic simple and straightforward. Avoid over-engineering the solution.**

## DOCUMENTATION:

Pandas documentation for data manipulation: https://pandas.pydata.org/docs/
Scikit-learn documentation for data splitting: https://scikit-learn.org/stable/modules/cross_validation.html

## OTHER CONSIDERATIONS:

- Follow the existing baseline module structure and conventions
- Use pandas for data manipulation as specified in CLAUDE.md
- Use scikit-learn for stratified sampling and cross-validation utilities
- Include proper error handling for edge cases (empty sites, missing data)
- Add comprehensive unit tests with at least 3 test cases per function
- Use type hints and pydantic for data validation
- Keep functions under 350 lines as per project guidelines
- Document all functions with Google-style docstrings
- Save split datasets in `results/baseline/` directory with proper naming conventions
- **CRITICAL: Prioritize simplicity over complexity - implement the minimum viable solution that meets requirements**

### Small Class Handling Strategies:
- Implement minimum sample size validation per class (e.g., minimum 5 samples per class)
- Provide warning/error handling for classes with insufficient samples
- Options for combining rare classes or excluding them from analysis
- Stratified sampling that preserves rare classes when possible
- Support for different sampling strategies: random, stratified, or balanced

### Train/Test Split Requirements:
- Random seed handling for reproducible splits
- Configurable split ratios with validation (e.g., 0.1 ≤ test_size ≤ 0.5)
- Stratification by outcome variables (COD, CSMF accuracy)
- Support for both simple random and stratified sampling
- Cross-validation strategies for imbalanced datasets (e.g., StratifiedKFold)
- Metadata tracking for split configurations and class distributions