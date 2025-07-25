# Workflow Pipeline - All Pending Tasks Automated Execution

This command orchestrates the complete development workflow from task planning to completion, automatically processing all pending tasks from TASK.md with no user intervention required.

## Pipeline Overview

```
START
  ↓
Read TASK.md and find all 📋 Planned tasks
  ↓
Sort tasks by priority and check dependencies
  ↓
For each pending task (in priority order):
  1. /next-tasks → Create NEXT_STEPS_TASKxx.md
  2. /generate-issue → Create GitHub issue & branch  
  3. /generate-prp → Create PRP from NEXT_STEPS
  4. /execute-prp → Implement the PRP
  5. /close-issue → Complete task & create PR
  ↓
Continue until all pending tasks are complete
  ↓
END
```

## Execution Steps

### Step 1: Initialize Pipeline

- Read TASK.md to find all tasks with 📋 Planned status
- Extract task details and dependencies for each pending task
- Create a list of executable tasks

### Step 2: Task Selection and Ordering

- Order tasks by priority: High → Medium → Low
- Check dependencies for each task
- Skip tasks with unmet dependencies
- Create execution queue based on priority and dependency status

### Step 3: Execute Task Pipeline

For each pending task in the execution queue:

**File Tracking Between Steps:**
- The pipeline maintains state between steps for each task
- Files generated in one step are passed to subsequent steps
- Example flow for task IM-052:
  - Step 3.1 generates: NEXT_STEPS_TASK52.md
  - Step 3.2 uses: /generate-issue NEXT_STEPS_TASK52.md
  - Step 3.3 uses: /generate-prp NEXT_STEPS_TASK52.md
  - Step 3.3 generates: PRPs/task-52-bootstrap-ci-fix.md
  - Step 3.4 uses: /execute-prp PRPs/task-52-bootstrap-ci-fix.md

#### 3.1 Planning Phase

Execute: `/next-tasks`

- Generate NEXT_STEPS_TASKxx.md based on:
  - @PLANNING.md (architecture and constraints)
  - @TASK.md (current progress and dependencies)
- Output: NEXT_STEPS_TASKxx.md where xx is the current task ID (e.g., NEXT_STEPS_TASK52.md for IM-052)

#### 3.2 Issue Creation

Execute: `/generate-issue {next_steps_file}`

- Update PLANNING.md and TASK.md
- Create GitHub issue with task details
- Create feature branch (e.g., feature/task-8-ai-matching)
- Checkout the new branch
- Note: {next_steps_file} is the actual file generated in step 3.1

#### 3.3 PRP Generation

Execute: `/generate-prp {next_steps_file}`

- Research codebase patterns
- Research external documentation
- Generate comprehensive PRP
- Output: PRPs/task-xx-[feature-name].md
- Note: {next_steps_file} is the same file from step 3.1

#### 3.4 Implementation

Execute: `/execute-prp {prp_file}`

- Load and understand PRP
- Create implementation plan with TodoWrite
- Execute all code changes
- Run validation commands
- Fix any issues until all tests pass
- Note: {prp_file} is the actual PRP file generated in step 3.3

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

After processing all pending tasks:

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

- Show current task being processed (e.g., "Task 3 of 7: IM-052")
- Display pipeline stage (1-5)
- Report validation results
- Log any errors or blockers
- Track completed vs remaining tasks

## Error Recovery

If any step fails:

- Stop at the failed step
- Report the error clearly
- Show which task failed and remaining tasks
- Provide recovery instructions
- Allow resuming from the failed task in the queue

## Fully Automated Execution

The pipeline runs all pending tasks without user checkpoints:

- Finds all tasks with 📋 Planned status
- Orders by priority and checks dependencies
- Executes all 5 pipeline steps for each task without pausing
- Automatically creates and merges pull requests
- Continues until all pending tasks are processed
- Only stops when all tasks are completed or an error occurs

## Example Usage

```
/workflow-pipeline-auto
```

This will:

1. Read TASK.md and find all 📋 Planned tasks
2. Order tasks by priority (High → Medium → Low)
3. Check dependencies for each task
4. Execute the complete pipeline for each task automatically (no user confirmation)
5. Create issues, branches, PRPs, implementations, and PRs for all tasks
6. Complete when all pending tasks are finished

## Notes

- Tasks are selected from TASK.md based on 📋 Planned status
- Priority ordering: High → Medium → Low
- Tasks with unmet dependencies are skipped and reconsidered later
- Each command in the pipeline is executed exactly as defined in its respective .md file
- The pipeline maintains state between steps and tasks
- All git operations are performed automatically
- Validation must pass before proceeding to next step
- **FULLY AUTOMATED**: No user confirmations or checkpoints - runs until all tasks complete
- User can interrupt the pipeline manually if needed, but it will not pause on its own
- The pipeline will automatically merge PRs for all completed tasks

## Critical Requirements

- **NEVER mark a task complete without real-world verification**
- **Unit tests alone are insufficient** - must test with actual project data
- **All model types must be tested** - partial functionality is not complete
- **Log files must be checked** - silent failures are still failures
- **Performance must be validated** - working slowly may indicate issues

The pipeline's primary goal is delivering working functionality, not just completing procedural steps.
