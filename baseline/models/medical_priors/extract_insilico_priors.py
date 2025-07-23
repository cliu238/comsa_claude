"""Extract InSilicoVA prior probability data from R package.

This script extracts the conditional probability matrices from the InSilicoVA
R package and saves them as CSV files for use in Python.

Requirements:
    - R must be installed
    - InSilicoVA R package must be installed (install.packages("InSilicoVA"))
"""

import subprocess
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def extract_insilico_priors(output_dir: Path):
    """Extract InSilicoVA prior data using R.
    
    Args:
        output_dir: Directory to save extracted CSV files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # R script to extract data
    r_script = """
    library(InSilicoVA)
    
    # Load the conditional probability data
    data(condprob)
    data(condprobnum)
    data(probbase)
    
    # Get dimensions
    cat("condprob dimensions:", dim(condprob), "\\n")
    cat("condprobnum dimensions:", dim(condprobnum), "\\n")
    
    # Convert to data frames with row/column names
    condprob_df <- as.data.frame(condprob)
    condprob_df$symptom <- rownames(condprob)
    
    condprobnum_df <- as.data.frame(condprobnum)
    condprobnum_df$symptom <- rownames(condprobnum)
    
    # Save as CSV files
    write.csv(condprob_df, "{output_dir}/condprob.csv", row.names = FALSE)
    write.csv(condprobnum_df, "{output_dir}/condprobnum.csv", row.names = FALSE)
    
    # Extract symptom and cause names
    symptoms <- rownames(condprob)
    causes <- colnames(condprob)
    
    write.csv(data.frame(symptom = symptoms), "{output_dir}/symptoms.csv", row.names = FALSE)
    write.csv(data.frame(cause = causes), "{output_dir}/causes.csv", row.names = FALSE)
    
    # If probbase exists as a separate object, save it
    if (exists("probbase")) {{
        if (is.matrix(probbase) || is.data.frame(probbase)) {{
            probbase_df <- as.data.frame(probbase)
            write.csv(probbase_df, "{output_dir}/probbase.csv", row.names = TRUE)
        }}
    }}
    
    cat("Data extraction complete\\n")
    """.format(output_dir=output_dir)
    
    # Save R script temporarily
    r_script_path = output_dir / "extract_data.R"
    with open(r_script_path, "w") as f:
        f.write(r_script)
    
    try:
        # Run R script
        result = subprocess.run(
            ["Rscript", str(r_script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("R script output:")
        logger.info(result.stdout)
        
        if result.stderr:
            logger.warning("R script warnings/errors:")
            logger.warning(result.stderr)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run R script: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("R or Rscript not found. Please install R and the InSilicoVA package.")
        raise
    finally:
        # Clean up R script
        if r_script_path.exists():
            r_script_path.unlink()
            
    logger.info(f"InSilicoVA prior data extracted to {output_dir}")


def convert_alphabetic_probs(condprob_df: pd.DataFrame) -> pd.DataFrame:
    """Convert alphabetic probability codes to numeric values.
    
    InSilicoVA uses alphabetic codes for probability levels:
    - A: Very likely (0.8-1.0)
    - B: Likely (0.5-0.8)
    - C: Possible (0.2-0.5)
    - D: Unlikely (0.05-0.2)
    - E: Very unlikely (0.01-0.05)
    - N: Never (0.0)
    - Y: Always (1.0)
    
    Args:
        condprob_df: DataFrame with alphabetic probability codes
        
    Returns:
        DataFrame with numeric probabilities
    """
    # Define mapping from alphabetic to numeric (using midpoints)
    prob_mapping = {
        'A': 0.9,    # Very likely
        'B': 0.65,   # Likely
        'C': 0.35,   # Possible
        'D': 0.125,  # Unlikely
        'E': 0.03,   # Very unlikely
        'N': 0.0,    # Never
        'Y': 1.0,    # Always
        '': 0.1,     # Default/missing
        '-': 0.1,    # Not applicable
    }
    
    # Convert each column except 'symptom'
    numeric_df = condprob_df.copy()
    for col in numeric_df.columns:
        if col != 'symptom':
            numeric_df[col] = numeric_df[col].map(
                lambda x: prob_mapping.get(str(x).upper().strip(), 0.1)
            )
            
    return numeric_df


def create_summary_stats(data_dir: Path):
    """Create summary statistics of the extracted data.
    
    Args:
        data_dir: Directory containing extracted CSV files
    """
    # Load data
    symptoms = pd.read_csv(data_dir / "symptoms.csv")
    causes = pd.read_csv(data_dir / "causes.csv")
    
    # Try to load numeric probabilities
    if (data_dir / "condprobnum.csv").exists():
        condprobnum = pd.read_csv(data_dir / "condprobnum.csv")
        logger.info("Using condprobnum (numeric values)")
    elif (data_dir / "condprob.csv").exists():
        # Convert alphabetic to numeric
        condprob = pd.read_csv(data_dir / "condprob.csv")
        condprobnum = convert_alphabetic_probs(condprob)
        condprobnum.to_csv(data_dir / "condprobnum_converted.csv", index=False)
        logger.info("Converted alphabetic probabilities to numeric")
    else:
        logger.error("No probability data found")
        return
        
    # Calculate statistics
    stats = {
        "n_symptoms": len(symptoms),
        "n_causes": len(causes),
        "matrix_shape": f"{len(symptoms)} x {len(causes)}",
        "top_10_symptoms": symptoms['symptom'].head(10).tolist(),
        "top_10_causes": causes['cause'].head(10).tolist()
    }
    
    # Save summary
    summary_path = data_dir / "data_summary.txt"
    with open(summary_path, "w") as f:
        f.write("InSilicoVA Prior Data Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
            
        f.write("\nFiles created:\n")
        for file in sorted(data_dir.glob("*.csv")):
            f.write(f"  - {file.name}\n")
            
    logger.info(f"Summary saved to {summary_path}")


def main():
    """Main function to extract InSilicoVA priors."""
    logging.basicConfig(level=logging.INFO)
    
    # Output directory
    output_dir = Path(__file__).parent / "data"
    
    try:
        # Extract data from R
        extract_insilico_priors(output_dir)
        
        # Create summary statistics
        create_summary_stats(output_dir)
        
        logger.info("Extraction complete!")
        logger.info(f"Data saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        
        # Create fallback message
        fallback_file = output_dir / "README.md"
        fallback_file.parent.mkdir(parents=True, exist_ok=True)
        with open(fallback_file, "w") as f:
            f.write("# InSilicoVA Prior Data\n\n")
            f.write("To extract InSilicoVA prior data, you need:\n\n")
            f.write("1. Install R: https://www.r-project.org/\n")
            f.write("2. Install InSilicoVA package in R:\n")
            f.write("   ```R\n")
            f.write("   install.packages('InSilicoVA')\n")
            f.write("   ```\n")
            f.write("3. Run the extraction script:\n")
            f.write("   ```bash\n")
            f.write("   python extract_insilico_priors.py\n")
            f.write("   ```\n\n")
            f.write("The script will extract:\n")
            f.write("- condprob.csv: Alphabetic conditional probabilities\n")
            f.write("- condprobnum.csv: Numeric conditional probabilities\n")
            f.write("- symptoms.csv: List of 245 symptoms\n")
            f.write("- causes.csv: List of 60 causes\n")
            f.write("- probbase.csv: Base probability table\n")


if __name__ == "__main__":
    main()