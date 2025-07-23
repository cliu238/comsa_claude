# Product Requirements Prompt (PRP) for Task RD-018

## Task Identification
**Task ID**: RD-018  
**Title**: Analyze XGBoost vs InSilicoVA Algorithmic Differences and Cross-Site Generalization  
**Type**: Research & Development (Markdown Report Only)  
**Priority**: High  
**Deliverable**: Comprehensive markdown report at `reports/xgboost_insilico_analysis.md`

## Context and Background

### Problem Statement
The VA34 site-based model comparison experiment (tasks IM-035 and IM-051) revealed a significant generalization gap between XGBoost and InSilicoVA models when applied across different VA sites. While XGBoost offers a 50x speed advantage, it suffers from a 37.7% performance drop in out-of-domain scenarios compared to InSilicoVA's 33.9% drop. In extreme cases, such as the Dar→Pemba transfer, XGBoost achieves only 3.3% accuracy, indicating catastrophic failure in cross-site generalization.

### Business Impact
Understanding and addressing this generalization gap is critical for:
- Deploying reliable VA models in diverse geographical settings
- Balancing computational efficiency with model robustness
- Informing future model selection and development strategies
- Improving health outcomes in resource-constrained environments

### Technical Context
- **XGBoost**: Gradient boosting framework using decision trees
- **InSilicoVA**: Bayesian hierarchical model with expert-informed priors
- **VA (Verbal Autopsy)**: Method for determining causes of death through structured interviews
- **CSMF Accuracy**: Cause-Specific Mortality Fraction accuracy metric
- **Domain Shift**: Systematic differences between training and deployment sites

## Research Objectives

### Primary Goals
1. **Identify Root Causes**: Understand the algorithmic differences that lead to InSilicoVA's superior generalization
2. **Quantify Impact**: Measure specific factors contributing to performance gaps
3. **Develop Solutions**: Propose concrete, implementable strategies to improve XGBoost's cross-site performance
4. **Inform Decision-Making**: Provide clear guidance for model selection in different deployment scenarios

### Research Questions
1. What algorithmic properties of InSilicoVA enable better cross-site generalization?
2. Which sites or site characteristics cause the most severe generalization failures?
3. How do feature importance patterns differ between models and across sites?
4. What domain adaptation techniques could improve XGBoost's robustness?
5. Can we identify "transferability indicators" that predict generalization success?

## Methodology Requirements

### Data Analysis Phase
1. **Load Experimental Results**
   - Import all CSV files from `results/full_va34_comparison_complete/`
   - Validate data completeness and consistency
   - Create derived metrics (performance ratios, relative drops, etc.)

2. **Statistical Analysis**
   - Calculate confidence intervals for performance differences
   - Perform significance testing on generalization gaps
   - Analyze variance components (site, model, interaction effects)

3. **Pattern Discovery**
   - Identify site clusters based on transfer performance
   - Detect systematic patterns in failure modes
   - Correlate performance with site characteristics

### Visualization Requirements

Create at least 8 high-quality visualizations using matplotlib/seaborn:

1. **Cross-Site Performance Heatmaps** (2 visualizations)
   - One for XGBoost, one for InSilicoVA
   - Sites on both axes, color intensity showing CSMF accuracy
   - Annotate with exact values for readability

2. **Generalization Gap Analysis** (2 visualizations)
   - Scatter plot: in-domain vs out-domain performance
   - Bar chart: ranked performance drops by site pair

3. **Site Characteristic Analysis** (2 visualizations)
   - PCA/t-SNE plot of sites based on transfer patterns
   - Correlation matrix of site features vs generalization success

4. **Model Comparison Dashboard** (1 visualization)
   - Multi-panel figure comparing key metrics side-by-side
   - Include performance, speed, and robustness dimensions

5. **Failure Mode Analysis** (1 visualization)
   - Focus on extreme cases (e.g., Dar→Pemba)
   - Show cause-specific performance breakdowns

### Algorithmic Analysis

1. **XGBoost Deep Dive**
   - Document tree construction algorithm
   - Analyze feature interaction learning
   - Examine regularization mechanisms (gamma, lambda, alpha)
   - Study prediction aggregation across trees

2. **InSilicoVA Deep Dive**
   - Explain Bayesian hierarchical structure
   - Document prior specification and expert knowledge integration
   - Analyze posterior computation and uncertainty quantification
   - Study conditional independence assumptions

3. **Comparative Analysis**
   - Create side-by-side comparison table
   - Identify fundamental philosophical differences
   - Map algorithmic properties to generalization capabilities
   - Analyze computational complexity trade-offs

### Literature Review Requirements

1. **Domain Adaptation in Healthcare**
   - Survey recent advances in medical AI generalization
   - Focus on tabular data methods (not just imaging)
   - Include both classical and deep learning approaches

2. **VA-Specific Literature**
   - Review successful multi-site VA studies
   - Document known challenges in VA generalization
   - Include WHO recommendations and standards

3. **Technical Solutions**
   - Instance reweighting techniques
   - Feature alignment methods
   - Ensemble approaches
   - Transfer learning for tabular data

## Report Structure and Content

### Executive Summary (1 page)
- Key findings in bullet points
- Visual abstract showing main results
- Clear recommendations for practitioners
- Cost-benefit analysis of different approaches

### 1. Introduction (2-3 pages)
- Problem motivation and impact
- Research objectives and scope
- Report structure overview

### 2. Experimental Setup Review (2 pages)
- VA34 dataset description
- Model configurations used
- Evaluation methodology
- Key metrics and their interpretation

### 3. Empirical Results Analysis (4-5 pages)
- Comprehensive performance comparison
- Site-specific analysis with visualizations
- Statistical significance testing
- Failure mode characterization

### 4. Algorithmic Deep Dive (5-6 pages)
- XGBoost mechanism analysis
- InSilicoVA mechanism analysis
- Comparative algorithmic properties
- Theoretical insights on generalization

### 5. Literature Synthesis (3-4 pages)
- Domain adaptation techniques
- Healthcare AI generalization
- VA-specific considerations
- Applicable solutions from other domains

### 6. Improvement Strategies (4-5 pages)
For each proposed strategy, include:
- Technical description
- Implementation complexity (Low/Medium/High)
- Expected performance impact
- Resource requirements
- Risk assessment

Minimum 5 strategies, such as:
1. Adversarial domain adaptation for XGBoost
2. Site-invariant feature engineering
3. Bayesian prior integration into gradient boosting
4. Multi-site ensemble methods
5. Active learning for site adaptation

### 7. Recommendations (2-3 pages)
- Decision tree for model selection
- Implementation roadmap
- Resource allocation guidance
- Future research directions

### 8. Conclusion (1 page)
- Summary of key insights
- Limitations of analysis
- Call to action

### Appendices
- Detailed statistical results
- Additional visualizations
- Code snippets for reproducibility
- Glossary of terms

## Technical Requirements

### Analysis Tools
- Python with pandas, numpy, scipy for data analysis
- Matplotlib and seaborn for visualizations
- Scikit-learn for additional ML analysis
- Markdown for report writing

### Data Sources
- Primary: `results/full_va34_comparison_complete/` directory
- Secondary: Algorithm documentation and papers
- Tertiary: Domain adaptation literature

### Quality Standards
- All visualizations must be publication-quality
- Statistical claims must include confidence intervals
- Recommendations must be actionable and specific
- Code snippets must be executable

## Success Criteria

### Quantitative Metrics
- [ ] Minimum 8 high-quality visualizations
- [ ] At least 5 concrete improvement strategies
- [ ] Statistical significance reported for all major claims
- [ ] Minimum 25 pages of comprehensive analysis

### Qualitative Metrics
- [ ] Report is accessible to both technical and medical audiences
- [ ] Clear narrative flow from problem to solution
- [ ] Actionable recommendations with implementation guidance
- [ ] Balanced perspective acknowledging trade-offs

### Deliverable Checklist
- [ ] Complete markdown report at `reports/xgboost_insilico_analysis.md`
- [ ] All visualizations saved as high-resolution PNG files
- [ ] Summary slides for stakeholder presentation
- [ ] List of future implementation tasks with priorities

## Constraints and Considerations

### Technical Constraints
- No code implementation in this task (research only)
- Must use existing experimental results
- Cannot rerun experiments due to time constraints

### Ethical Considerations
- Acknowledge limitations of automated VA systems
- Consider equity implications of model choice
- Discuss deployment considerations for low-resource settings

### Timeline
- Total effort: 5 days
- Data analysis: 2 days
- Algorithmic research: 2 days
- Report writing and revision: 1 day

## Future Work

Based on this research, create GitHub issues for:
- [IM-052] Implement domain adaptation for XGBoost VA models
- [IM-053] Create ensemble methods combining XGBoost and InSilicoVA
- [IM-054] Develop site-invariant feature engineering pipeline
- [IM-055] Build automated model selection framework
- [IM-056] Create cross-site validation toolkit

## Additional Notes

### Key Stakeholders
- VA researchers and practitioners
- Public health officials in target regions
- ML engineers implementing VA systems
- Policy makers funding VA initiatives

### Communication Guidelines
- Use clear, jargon-free language where possible
- Define all technical terms on first use
- Include visual aids to support complex concepts
- Provide executive-friendly summaries

### Research Integrity
- Acknowledge all data sources and prior work
- Report both positive and negative findings
- Discuss limitations transparently
- Avoid overstatement of results

This PRP provides comprehensive guidance for completing task RD-018. The research should produce actionable insights that bridge the gap between XGBoost's computational efficiency and InSilicoVA's superior generalization, ultimately improving VA model deployment in diverse global health settings.