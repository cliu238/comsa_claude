## FEATURE:

- Baseline VA (Verbal Autopsy) data processing pipeline that uses the va-data submodule to process PHMRC data
- Integration with JH-DSAI/va-data GitHub repository as a submodule for standardized VA data processing
- Data validation and preprocessing stage that handles PHMRC CSV files with appropriate transformations
- Support for both numeric (va34) and text (gs_text34) target formats with configurable encoding options

## EXAMPLES:

In the `examples/` folder, there is data validation code that demonstrates the core processing logic:

- `examples/data_validation.py` - Reference implementation showing how to validate and transform VA data using PHMRCData class
- Uses va_data.va_data_core for PHMRC data handling with OpenVA transformations
- Implements both standard pipeline processing and Table 3 compatible preprocessing modes
- Handles categorical to numeric conversions for ML model compatibility

The ml_pipeline repository (https://github.com/JH-DSAI/ml_pipeline) the stage-based architecture pattern provide reference:

- `ml_pipeline/stages/data_validation.py` - Stage-based data validation approach to follow

## DOCUMENTATION:

- VA Data Core Documentation: https://github.com/JH-DSAI/va-data
- PHMRC Data Source: `data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv`
- ML Pipeline Architecture: https://github.com/JH-DSAI/ml_pipeline

## IMPLEMENTATION REQUIREMENTS:

### Data Processing Pipeline

- Load PHMRC CSV data using PHMRCData class from va-data submodule
- Validate data integrity and handle missing values appropriately
- Support OpenVA encoding transformation for InSilicoVA compatibility
- Convert categorical features to numeric format for ML models
- Handle both va34 (numeric) and gs_text34 (text) target formats

### Architecture Components

- `baseline/data/data_loader_preprocessor.py` - PHMRC data loading and initial validation, Feature transformation and encoding logic
- `baseline/config/data-config.py` - Configuration for data paths and processing options

### Key Features

- Configurable preprocessing modes (standard vs Table 3 compatible)
- Site-based data stratification support
- Automatic handling of VA-specific data patterns (Yes/No, Y/N mappings)
- Robust missing value handling with appropriate defaults
- Feature column filtering (exclude metadata columns)

## OTHER CONSIDERATIONS:

- Add va-data as git submodule: `git submodule add https://github.com/JH-DSAI/va-data`
- Ensure proper virtual environment setup with va-data dependencies
- Include comprehensive unit tests for data validation stages
- Document expected data format and column specifications
- Implement progress logging for long-running data processing
- Save processed data in `results/baseline/` directory structure
- Use Poetry for dependency management with appropriate package versions
- Follow KISS and YAGNI principles - avoid over-engineering the pipeline
