## FEATURE:

- Simple, clean data split module for the baseline package that splits VA data by site column
- Integration with existing baseline configuration system (DataConfig) 
- Support for multiple split strategies: train/test, cross-site validation, and stratified splits
- Designed for future data analysis workflows with clear, reusable interface
- Minimal complexity compared to existing examples that are overcomplicated

## EXAMPLES:

In the `examples/` folder, there are existing splitting implementations that you can refer to, but they are overcomplicated and not suitable for direct use:

- `examples/cross_site_splitting.py` - Reference for cross-site logic but simplify the multi-scenario approach. Remove complex nested dictionary structures and hardcoded scenarios.
- `examples/dataset_splitting_v2.py` - Reference for stratified splitting but remove manual shuffling complexity and single-instance class handling.
- `baseline/data/data_loader_preprocessor.py` - Follow the same clean architecture pattern with proper logging and configuration integration.
- `baseline/config/data_config.py` - Use similar Pydantic configuration approach for split settings.

Don't copy any of these examples directly, as they contain unnecessary complexity. Use them as reference for the concepts but create a much simpler implementation.

## DOCUMENTATION:

- Scikit-learn documentation for train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Pandas documentation for DataFrame operations: https://pandas.pydata.org/docs/
- Pydantic documentation for configuration: https://docs.pydantic.dev/

## OTHER CONSIDERATIONS:

- Follow the existing baseline module structure and patterns
- Extend DataConfig with split-related configuration options
- Create `baseline/data/data_splitter.py` with a clean `VADataSplitter` class
- Support three core split modes:
  1. **Simple train/test split** - Standard sklearn train_test_split by site
  2. **Cross-site validation** - Train on some sites, test on others  
  3. **Stratified site split** - Maintain label distribution within each site
- Include comprehensive unit tests following existing test patterns in `tests/baseline/`
- Add example usage script demonstrating all split modes
- No complex file I/O - return DataFrames for flexibility in future analysis
- Use python_dotenv and load_env() for environment variables if needed
- Follow CLAUDE.md principles: KISS (Keep It Simple), YAGNI (You Aren't Gonna Need It), modularity
- Ensure >90% test coverage and integration with existing baseline module