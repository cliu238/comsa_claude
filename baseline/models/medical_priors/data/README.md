# InSilicoVA Prior Data

This directory should contain the conditional probability matrices extracted from the InSilicoVA R package.

## Required Files

The following files should be present after extraction:
- `condprob.csv` - Alphabetic conditional probabilities (245 symptoms × 60 causes)
- `condprobnum.csv` - Numeric conditional probabilities (0-1 values)
- `symptoms.csv` - List of symptom names
- `causes.csv` - List of cause names
- `probbase.csv` - Base probability table (if available)

## How to Extract Data

### Prerequisites
1. Install R from https://www.r-project.org/
2. Install the InSilicoVA package in R:
   ```R
   install.packages('InSilicoVA')
   ```

### Extraction Process
Run the extraction script from the parent directory:
```bash
cd baseline/models/medical_priors
python extract_insilico_priors.py
```

This will:
1. Load the InSilicoVA R package
2. Extract the conditional probability matrices
3. Save them as CSV files in this directory
4. Create a summary of the extracted data

## Data Structure

### Conditional Probability Matrix
- **Rows**: 245 symptoms (e.g., fever, cough, chest pain)
- **Columns**: 60 causes of death (e.g., HIV/AIDS, malaria, tuberculosis)
- **Values**: Probability of symptom given cause P(symptom|cause)

### Alphabetic Probability Codes (condprob)
- `A`: Very likely (0.8-1.0)
- `B`: Likely (0.5-0.8)
- `C`: Possible (0.2-0.5)
- `D`: Unlikely (0.05-0.2)
- `E`: Very unlikely (0.01-0.05)
- `N`: Never (0.0)
- `Y`: Always (1.0)

### Numeric Probabilities (condprobnum)
Direct probability values between 0 and 1.

## Using the Data

The prior loader (`prior_loader.py`) will automatically:
1. Check for these CSV files
2. Load them into memory
3. Create the `MedicalPriors` data structure
4. Fall back to simulated priors if files are not found

## Alternative: Using Simulated Priors

If you cannot extract the real InSilicoVA data, the system will use simulated priors that approximate the structure and relationships found in real VA data. While not as accurate as the real priors, they demonstrate the functionality of the prior integration system.