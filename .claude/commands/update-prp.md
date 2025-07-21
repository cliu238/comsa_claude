# Update PRP

## PRP File: $ARGUMENTS

Iteratively improve an existing PRP based on validation feedback, execution results, or new requirements. This ensures PRPs evolve and improve over time.

## Update Process

### 1. **Initial Assessment**
   - Load the current PRP file
   - Run validation to identify current issues
   - Check for any execution history or feedback
   - Determine update priorities

### 2. **Gap Analysis**
   Identify what needs improvement:
   - **Missing Context**: Documentation, examples, gotchas
   - **Unclear Instructions**: Ambiguous tasks, vague goals  
   - **Weak Validation**: Insufficient tests, missing checks
   - **Integration Issues**: Incomplete connection points
   - **Error Handling**: Unaddressed edge cases

### 3. **Research Phase**
   Fill identified gaps:
   - **Documentation Search**: Find missing API docs, guides
   - **Code Analysis**: Locate better pattern examples
   - **Error Patterns**: Research common failures
   - **Best Practices**: Current recommended approaches
   - **Version Updates**: Check for API/library changes

### 4. **Enhancement Implementation**
   Update PRP sections systematically:
   
   **Context Enhancement**
   ```yaml
   # ADD missing references
   - url: [New documentation URL]
     why: [Specific value it provides]
     critical: [Key insight discovered]
   
   # UPDATE outdated information
   - file: [Updated path if moved]
     why: [Better example found]
   ```
   
   **Implementation Clarification**
   - Add detailed pseudocode where vague
   - Include specific error handling patterns
   - Clarify task dependencies and ordering
   - Add integration test scenarios
   
   **Validation Strengthening**
   - Add missing test cases
   - Include edge case handling
   - Specify exact expected outputs
   - Add troubleshooting steps

### 5. **Quality Improvements**
   Based on validation scoring:
   
   **Low Context Score (< 2/3)**
   - Add comprehensive documentation links
   - Include more code examples
   - Document all external dependencies
   - Add library-specific gotchas
   
   **Low Implementation Score (< 2/3)**
   - Break down complex tasks
   - Add step-by-step pseudocode
   - Clarify data flow
   - Specify exact file modifications
   
   **Low Validation Score (< 2/2)**
   - Add unit test examples
   - Include integration tests
   - Specify exact commands
   - Add error recovery steps
   
   **Low Error Handling Score (< 2/2)**
   - Document known failure modes
   - Add try-catch patterns
   - Include timeout handling
   - Specify rollback procedures

### 6. **Version Management**
   - Save original as: `PRPs/archive/[name]_v[timestamp].md`
   - Update main file with improvements
   - Add version header with changes
   - Update confidence score

## Update Strategies

### For Failed Executions
1. Analyze failure points from execution logs
2. Add missing context that caused confusion
3. Clarify ambiguous instructions
4. Strengthen validation for failure points
5. Add specific troubleshooting steps

### For Partial Success
1. Identify what worked well (preserve)
2. Enhance sections that struggled
3. Add integration tests for connections
4. Improve error handling examples
5. Update success criteria clarity

### For New Requirements
1. Add new goals/features cleanly
2. Update task ordering if needed
3. Add new validation criteria
4. Ensure backward compatibility
5. Document requirement changes

## Output Format

### Updated PRP Header
```markdown
---
name: "[Original Name] - Updated"
version: "2.0"
updated: "[timestamp]"
confidence: "[new score]/10 (was [old score]/10)"
changes: |
  - Added missing Gmail API authentication docs
  - Clarified task ordering for database setup
  - Added edge case handling for rate limits
  - Strengthened validation with integration tests
---
```

### Change Summary Report
```markdown
# PRP Update Summary: [filename]

## Version: 1.0 → 2.0
## Confidence: [old]/10 → [new]/10

### Major Improvements
1. **[Section]**: [What was added/changed]
   - Impact: [How this helps implementation]

### Context Additions
- Added [X] new documentation references
- Included [Y] code examples
- Documented [Z] gotchas

### Validation Enhancements
- Added [N] test cases
- Strengthened [type] validation
- Included troubleshooting for [scenarios]

### Next Recommended Updates
- [ ] [Future improvement opportunity]
```

## Best Practices for Updates

1. **Preserve What Works**: Don't change successful patterns
2. **Incremental Improvement**: Focus on biggest gaps first
3. **Test Updates**: Validate the updated PRP before use
4. **Document Changes**: Clear version history helps learning
5. **Learn from Failures**: Each update should prevent past issues

## Common Update Patterns

### Adding Missing Context
```yaml
# BEFORE: Vague reference
- See API docs for authentication

# AFTER: Specific and actionable  
- url: https://developers.google.com/gmail/api/auth/scopes
  why: Required OAuth2 scopes for draft creation
  critical: Use 'gmail.compose' scope, not 'gmail.send'
```

### Clarifying Tasks
```yaml
# BEFORE: High-level task
Task 1: Implement authentication

# AFTER: Detailed steps
Task 1: Implement Gmail OAuth2 authentication
  - CREATE src/auth/gmail_auth.py
  - PATTERN: Follow src/auth/oauth_base.py structure
  - MODIFY: Use Gmail-specific scopes
  - STORE: Tokens in secure keyring (see src/auth/token_storage.py)
  - TEST: Verify with test_gmail_auth.py template
```

### Strengthening Validation
```bash
# BEFORE: Basic test
pytest tests/

# AFTER: Comprehensive validation
# Unit tests with coverage
pytest tests/unit/ -v --cov=src --cov-report=term-missing

# Integration test for Gmail API
pytest tests/integration/test_gmail_draft.py -v -s

# Expected: All tests pass, >80% coverage
# If fails: Check logs/test_errors.log for API response codes
```

Remember: Each update should measurably improve the PRP's success rate for one-pass implementation.