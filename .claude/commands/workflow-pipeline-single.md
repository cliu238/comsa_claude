# Workflow Pipeline - Single Task Automated Execution

This command orchestrates the complete development workflow from task planning to completion, automatically processing the specified task ($ARGUMENTS1) from TASK.md with no user intervention required.

## Pipeline Overview

```
START
  ↓
Validate specified task from $ARGUMENTS1
  ↓
For the specified task:
  1. /next-tasks → Create NEXT_STEPS_TASKxx.md
  2. /generate-issue → Create GitHub issue & branch  
  3. /generate-prp → Create PRP from NEXT_STEPS
  4. /execute-prp → Implement the PRP
  5. /close-issue → Complete task & create PR
  ↓
Complete single task
  ↓
END
```

## Execution Steps

### Step 1: Initialize Pipeline

- Read TASK.md to validate specified task ($ARGUMENTS1)
- Verify task exists and is in 📋 Planned status
- Extract task details and dependencies

### Step 2: Task Validation

- Validate the specified task ID (e.g., IM-035)
- Extract from the task:
  - Task number and ID
  - Task name and description
  - Dependencies and requirements
  - Priority level

### Step 3: Execute Task Pipeline

For the specified task:

#### 3.1 Planning Phase

Execute: `/new-tasks`

- Generate NEXT_STEPS_TASKxx.md based on:
  - @PLANNING.md (architecture and constraints)
  - @TASK.md (current progress and dependencies)
- Output: NEXT_STEPS_TASKxx.md with detailed planning

#### 3.2 Issue Creation

Execute: `/generate-issue NEXT_STEPS_TASKxx.md`

- Update PLANNING.md and TASK.md
- Create GitHub issue with task details
- Create feature branch (e.g., feature/task-8-ai-matching)
- Checkout the new branch

#### 3.3 PRP Generation

Execute: `/generate-prp NEXT_STEPS_TASKxx.md`

- Research codebase patterns
- Research external documentation
- Generate comprehensive PRP
- Output: PRPs/task-xx-[feature-name].md

#### 3.4 Implementation

Execute: `/execute-prp PRPs/task-xx-[feature-name].md`

- Load and understand PRP
- Create implementation plan with TodoWrite
- Execute all code changes
- Run validation commands
- Fix any issues until all tests pass

#### 3.4.1 Real-World Verification (CRITICAL)

Before marking implementation complete:

1. **Test with actual data** - Run the implementation with real project data:
   - Use existing processed data files if available
   - Test all major code paths and model types
   - Verify output files are created correctly
   - Check logs for any errors or warnings

2. **Verify all components work** - For ML/VA tasks:
   - Test each model type independently (XGBoost, InSilicoVA, etc.)
   - Confirm data format compatibility for each model
   - Ensure parallel execution works if implemented
   - Validate results make sense (no 0% accuracy, etc.)

3. **Verify with subagent data-root-cause-analyst** - Use the data-root-cause-analyst subagent to:
   - Analyze any unexpected results or performance issues
   - Diagnose potential data quality or algorithmic problems
   - Design experiments to validate hypotheses about issues
   - **If the agent identifies any problems, MUST fix them before proceeding**

4. **Document test commands** - Save the exact commands used for testing:
   - Include in PR description
   - Update README if new usage patterns

5. **Fix all failures** - Do NOT proceed if:
   - Any model fails to run
   - Errors appear in logs
   - Output is missing or incorrect
   - Performance is significantly degraded
   - The data-root-cause-analyst identifies unresolved issues

Only after ALL real-world tests pass and the data-root-cause-analyst confirms no issues should the task continue to completion.

#### 3.5 Completion

Execute: `/close-issue`

- Update TASK.md (move to completed, add notes)
- Commit all changes
- Create pull request with "Closes #X"
- Merge PR
- Delete feature branch
- Return to main branch

### Step 4: Final Verification & Completion

Before marking task as complete:

1. **Verification Checklist**:
   - ✓ All tests pass (unit tests, integration tests)
   - ✓ Real-world data tests successful (no model failures)
   - ✓ All intended functionality works as designed
   - ✓ Performance meets requirements
   - ✓ Documentation updated (README, usage examples)
   - ✓ No errors in logs during real usage

2. **Completion Actions**:
   - Update TASK.md status only after verification
   - Report successful completion with:
     - Task details
     - Test results summary
     - Commands to reproduce tests
   - Include output locations and usage instructions
   - Update README.md with new functionality

3. **Exit Criteria**:
   - Task marked complete ONLY when fully functional
   - All verification steps documented in PR
   - Real-world usage confirmed working

## Progress Tracking

Throughout execution:

- Show current task being processed
- Display pipeline stage (1-5)
- Report validation results
- Log any errors or blockers

## Error Recovery

If any step fails:

- Stop at the failed step
- Report the error clearly
- Provide recovery instructions
- Allow resuming from the failed step

## Fully Automated Execution

The pipeline runs the specified task without user checkpoints:

- Validates the task ID provided in $ARGUMENTS1
- Executes all 5 pipeline steps without pausing
- Automatically creates and merges pull request
- Completes when the single task is finished
- Only stops when task is completed or an error occurs

## Example Usage

```
/workflow-pipeline-part IM-035
```

This will:

1. Validate that task IM-035 exists in TASK.md
2. Verify the task is in 📋 Planned status
3. Execute the complete pipeline for that specific task automatically (no user confirmation)
4. Create issue, branch, PRP, implementation, and PR
5. Complete when the single task is finished

## Notes

- Task ID must be provided as the first argument (e.g., IM-035, IM-046)
- Each command in the pipeline is executed exactly as defined in its respective .md file
- The pipeline maintains state between steps
- All git operations are performed automatically
- Validation must pass before proceeding to next step
- **FULLY AUTOMATED**: No user confirmations or checkpoints - runs until task completion
- User can interrupt the pipeline manually if needed, but it will not pause on its own
- The pipeline will automatically merge the PR for the specified task

## Critical Requirements

- **NEVER mark a task complete without real-world verification**
- **Unit tests alone are insufficient** - must test with actual project data
- **All model types must be tested** - partial functionality is not complete
- **Log files must be checked** - silent failures are still failures
- **Performance must be validated** - working slowly may indicate issues

The pipeline's primary goal is delivering working functionality, not just completing procedural steps.
