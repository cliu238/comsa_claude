## FEATURE:

- Reproducible pipeline for evaluating and improving cause-of-death assignment from verbal-autopsy (VA) data
- Three-part architecture: Baseline Benchmark, Transfer Learning, and Active Learning
- Using PHMRC gold-standard dataset as starting point, expanding to COMSA data via transfer and active learning
- Poetry for dependency management with modular project structure

## EXAMPLES:

Data samples available in `data/`:

- `data/raw/PHMRC/` - PHMRC gold standard VA data (adult, child, neonate)
- `data/raw/MITS/` - MITS VA data files
- `data/raw/COMSA/` - COMSA WHO-2016 VA files
- `data/raw/who/` - WHO VA standard files

## DOCUMENTATION:

- OpenVA library: https://github.com/verbal-autopsy-software/openVA
- VA data preprocessing: https://github.com/JH-DSAI/va-data

## OTHER CONSIDERATIONS:

- Poetry is used for dependency management (pyproject.toml exists)
- Data files are stored in root `data/` folder with raw subdirectory structure
- Docker environment provided for InSilicoVA/InterVA algorithms (see `models/insilico/Dockerfile`)
- Each pipeline part (baseline, transfer, active) should have separate markdown documentation
- Use stratified splits by site & age-group for train/test
- 5-fold CV for hyperparameter tuning
- Key metrics: CSMF accuracy, Top-1/Top-3 COD accuracy, COD5
- Deliverables include benchmark_results.csv for baseline evaluation
