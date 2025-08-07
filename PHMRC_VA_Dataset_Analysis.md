# PHMRC Verbal Autopsy Dataset Analysis

## Dataset Overview

The PHMRC (Population Health Metrics Research Consortium) Verbal Autopsy dataset consists of three age-stratified files containing validated verbal autopsy data with gold standard cause of death assignments. This is a benchmark dataset widely used for developing and evaluating verbal autopsy algorithms.

### File Structure
- **Adult Dataset**: `IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv` (7,841 records)
- **Child Dataset**: `IHME_PHMRC_VA_DATA_CHILD_Y2013M09D11_0.csv` (2,064 records)  
- **Neonate Dataset**: `IHME_PHMRC_VA_DATA_NEONATE_Y2013M09D11_0.csv` (2,625 records)
- **Codebook**: `IHME_PHMRC_VA_DATA_CODEBOOK_Y2013M09D11_0.xlsx` (binary Excel file)

**Total Records**: 12,530 cases across all age groups

## Column Structure Analysis

### Primary Focus: Adult Dataset (946 columns)

The adult dataset contains 946 variables organized into distinct categories:

## 1. LABEL/TARGET COLUMNS

### Gold Standard Cause of Death (Primary Labels)
- **gs_code34** (Column 3): ICD-10 code for cause of death
- **gs_text34** (Column 4): Text description of cause of death 
- **va34** (Column 5): Numeric cause ID for 34-cause list

### Secondary Label Columns
- **gs_code46** (Columns 6-8): 46-cause classification
- **gs_code55** (Columns 9-11): 55-cause classification  
- **gs_comorbid1/gs_comorbid2** (Columns 12-13): Comorbidity information
- **gs_level** (Column 14): Gold standard level indicator

**Recommendation**: Use **gs_text34** as the primary target variable for cause of death prediction. This provides a 34-cause classification system which is the standard for VA algorithm evaluation.

### Cause Distribution (Adult Dataset)
Top causes by frequency:
1. Stroke (630 cases, 8.0%)
2. Other Non-communicable Diseases (599 cases, 7.6%)
3. Pneumonia (540 cases, 6.9%)
4. AIDS (502 cases, 6.4%)
5. Maternal (468 cases, 6.0%)

Total of 34 distinct causes of death represented.

## 2. FEATURE COLUMNS FOR PREDICTION

### A. Demographic Information (Columns 15-71)
- **Birth/Death dates**: g1_01d/m/y, g1_06d/m/y
- **Sex**: g1_05 
- **Age information**: Various age-related fields
- **Marital status, education**: g1_07a/b/c, g1_08-10
- **Respondent information**: g2_*, g3_*, g4_*, g5_* sections

### B. Symptom Indicators (Columns 72-266) 
**Primary symptom features - CRITICAL for prediction**

#### Symptom Categories:
- **a1_01_1 to a1_01_14**: Initial symptom screening (14 variables)
- **a2_01 to a2_87_10b**: Detailed symptom inquiry (188 variables)
- **a3_01 to a3_20**: Additional symptom details (20 variables) 
- **a4_01 to a4_06**: Healthcare utilization (6 variables)
- **a5_01 to a5_04**: Treatment history (9 variables)
- **a6_01 to a6_10**: Final illness characteristics (25 variables)
- **a7_11 to a7_14**: Death circumstances (4 variables)

**Data Types**: 
- Categorical: "Yes"/"No"/"Don't Know"
- Numeric: Duration values, counts
- Text: Some free-text responses

### C. Narrative Word Count Features (Columns 267-945)
**680 word count variables**: word_abdomen, word_abl, word_accid, etc.

These represent frequency counts of specific words appearing in the narrative text portions of the verbal autopsy interview. Values are typically small integers (0-5).

**Examples**:
- word_heart, word_chest, word_pain for cardiovascular symptoms
- word_fever, word_cough, word_breath for infectious diseases
- word_cancer, word_tumor for malignancies

## 3. COLUMNS TO DROP/EXCLUDE

### Identifiers & Metadata (Not for prediction):
- **module** (Column 2): VA module version
- **newid** (Column 946): Unique record identifier
- **gs_level** (Column 14): Administrative metadata


### Redundant Label Columns:
Keep only primary cause (gs_text34), drop:
- va34, gs_code34 (redundant encodings)
- 46-cause and 55-cause variants
- gs_comorbid1/gs_comorbid2 (secondary information)

## 4. DATA QUALITY CONSIDERATIONS

### Missing Data Patterns:
- "Don't Know" responses are common and meaningful
- Some fields have empty values
- Need proper encoding for categorical variables

### Preprocessing Requirements:
1. **Handle "Don't Know"**: Treat as separate category, not missing
2. **Date Processing**: Convert dates to age or time intervals. Specific birth/death dates should be converted to age:
3. **Categorical Encoding**: One-hot encode Yes/No/Don't Know variables
4. **Feature Scaling**: Normalize word count features
5. **Class Imbalance**: Address unequal cause distribution

## 5. RECOMMENDED FEATURE SETS

### Minimal Feature Set (High Signal):
- Key symptom indicators (a2_* series most important)
- Basic demographics (sex, age)
- Selected word features related to specific symptoms

### Full Feature Set:
- All symptom columns (a1_* through a7_*)
- Processed demographic variables
- All word count features
- Healthcare utilization indicators

### Feature Engineering Opportunities:
1. **Symptom Combinations**: Create composite features
2. **Age-Sex Interactions**: Important for cause-specific patterns
3. **Word Feature Aggregation**: Group related medical terms
4. **Duration Variables**: Extract temporal patterns from illness progression

## 6. VALIDATION CONSIDERATIONS

### Cross-Site Validation:
- **site** variable enables geographic validation
- Different sites may have different symptom patterns
- Consider site as stratification variable

### Age-Specific Models:
- Adult, Child, Neonate datasets require separate models
- Different cause distributions and symptom patterns
- Age-specific validation important

## Summary Recommendations

**Target Variable**: `gs_text34` (34-cause classification)

**Core Features**:
- Symptom indicators (columns 72-266): Primary predictive features
- Word count features (columns 267-945): Supplementary narrative information
- Selected demographics (processed age, sex)

**Drop**:
- Identifier columns (newid, module)
- Raw date columns
- Redundant label encodings
- Administrative metadata

**Special Handling**:
- "Don't Know" as legitimate category
- Class imbalance mitigation
- Site-stratified validation
- Age-appropriate modeling

This dataset provides a rich foundation for developing verbal autopsy algorithms with gold standard validation, making it ideal for benchmarking cause of death prediction models.