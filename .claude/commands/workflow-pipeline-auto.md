# Workflow Pipeline - Fully Automated Continuous Task Execution

This command orchestrates the complete development workflow from task planning to completion, automatically processing all pending tasks in TASK.md with no user intervention required.

## Pipeline Overview

```
START
  ↓
Check TASK.md for pending tasks
  ↓
For each pending task:
  1. /next-tasks → Create NEXT_STEPS_TASKxx.md
  2. /generate-issue → Create GitHub issue & branch  
  3. /generate-prp → Create PRP from NEXT_STEPS
  4. /execute-prp → Implement the PRP
  5. /close-issue → Complete task & create PR
  ↓
LOOP until all tasks completed
  ↓
END
```

## Execution Steps

### Step 1: Initialize Pipeline

- Read TASK.md to identify pending tasks
- Check current sprint status
- List all High/Medium/Low priority tasks

### Step 2: Task Selection

- Process tasks in priority order: High → Medium → Low
- For each task, extract:
  - Task number (e.g., Task 8, Task 9)
  - Task name and description
  - Any specific requirements

### Step 3: Execute Task Pipeline

For each selected task:

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
- run it at least once to make sure it all works(IMPORTANT!!!!)

#### 3.5 Completion

Execute: `/close-issue`

- Update TASK.md (move to completed, add notes)
- Commit all changes
- Create pull request with "Closes #X"
- Merge PR
- Delete feature branch
- Return to main branch

### Step 4: Loop Control

- After completing a task, check for more pending tasks
- If tasks remain, continue with next priority task
- If all tasks completed, exit pipeline
- If the whole loop has to stop or pause somewhere, please keep the time stamp
- Report how to run everything or where is the output/result and update README.md

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

The pipeline runs continuously without user checkpoints:

- Automatically selects the highest priority pending task
- Executes all 5 pipeline steps without pausing
- Automatically creates and merges pull requests
- Continues to the next task immediately after completion
- Only stops when all tasks are completed or an error occurs

## Example Usage

```
/workflow-pipeline
```

This will:

1. Analyze TASK.md for all pending tasks
2. Start with the highest priority task
3. Execute the complete pipeline for that task automatically (no user confirmation)
4. Continue with the next task immediately
5. Repeat until all tasks are completed without stopping

## Notes

- Each command in the pipeline is executed exactly as defined in its respective .md file
- The pipeline maintains state between steps
- All git operations are performed automatically
- Validation must pass before proceeding to next step
- **FULLY AUTOMATED**: No user confirmations or checkpoints - runs continuously until completion
- User can interrupt the pipeline manually if needed, but it will not pause on its own
- The pipeline will automatically merge PRs and continue to the next task
