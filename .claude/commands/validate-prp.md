# Validate PRP

## PRP File: $ARGUMENTS

Validate a PRP (Product Requirements Prompt) file for completeness, accuracy, and quality before execution. This helps ensure higher success rates for one-pass implementation.

## Validation Process

### 1. **Load and Parse PRP**
   - Read the specified PRP file
   - Verify it exists and is properly formatted
   - Extract all sections for analysis

### 2. **Structural Validation**
   Check for required sections:
   - [ ] Goal - Clear and specific
   - [ ] Why - Business value explained
   - [ ] What - User-visible behavior defined
   - [ ] Success Criteria - Measurable outcomes
   - [ ] Context - Documentation and references
   - [ ] Implementation Blueprint - Clear approach
   - [ ] Validation Loop - Executable tests
   - [ ] Final Checklist - Completion criteria

### 3. **Context Validation**
   Verify all references:
   - **Documentation URLs**: Check if accessible (use WebFetch to verify)
   - **File References**: Confirm files exist in codebase
   - **Code Examples**: Validate syntax (no obvious errors)
   - **Dependencies**: Ensure all are documented

### 4. **Implementation Clarity**
   Assess blueprint quality:
   - Task ordering is logical
   - Pseudocode is clear and detailed
   - Integration points are specified
   - Error handling is addressed
   - Patterns to follow are identified

### 5. **Validation Gates**
   Check executable commands:
   - Linting commands use correct syntax
   - Test commands reference right paths
   - Expected outputs are defined
   - Fix strategies are included

### 6. **Quality Scoring (1-10)**
   
   **Context Completeness (0-3 points)**
   - 3: All docs, examples, gotchas included
   - 2: Most context present, minor gaps
   - 1: Some context, significant gaps
   - 0: Minimal context provided

   **Implementation Clarity (0-3 points)**
   - 3: Crystal clear path, detailed pseudocode
   - 2: Good structure, some ambiguity
   - 1: Basic structure, needs clarification
   - 0: Unclear implementation approach

   **Validation Robustness (0-2 points)**
   - 2: Comprehensive tests, clear fix paths
   - 1: Basic validation, some gaps
   - 0: Minimal or missing validation

   **Error Handling (0-2 points)**
   - 2: Gotchas documented, patterns clear
   - 1: Some error cases covered
   - 0: No error handling guidance

## Output Format

### Validation Report
```markdown
# PRP Validation Report: [filename]

## Overall Score: X/10

### Strengths
- [What's done well]

### Issues Found
1. **[Issue Category]**: [Specific problem]
   - Impact: [How this affects implementation]
   - Fix: [How to address it]

### Missing Elements
- [ ] [Missing documentation/context]
- [ ] [Missing validation steps]

### Recommendations
1. [Specific improvement action]
2. [Additional context to add]

### Confidence Assessment
Based on this validation, the estimated success rate for one-pass implementation is: [X]%
```

## Validation Execution

1. First, check if the PRP file exists
2. Parse and analyze each section
3. Verify external references (sample a few URLs)
4. Check code examples for basic syntax
5. Score each category
6. Generate detailed report with actionable feedback

## Common Issues to Check

- **Vague Goals**: "Build a feature" vs "Build X that does Y for users Z"
- **Missing Gotchas**: Library quirks, version issues, common errors
- **Weak Validation**: "Run tests" vs specific test commands with expected output
- **Assumed Context**: References to patterns without showing examples
- **Incomplete Tasks**: High-level tasks without implementation details

## Anti-Patterns to Flag

- ❌ No error handling documented
- ❌ Validation commands that won't run
- ❌ Missing critical dependencies
- ❌ Circular or unclear task ordering
- ❌ No success criteria defined
- ❌ Context without explanation of relevance

Remember: The goal is to identify issues BEFORE execution to ensure smooth implementation.