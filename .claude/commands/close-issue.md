# Close Issue Command

## Steps to close an issue after task completion:

1. **UPDATE TASK.md FIRST** (REQUIRED):
   - Move task from "In Progress" to "Completed Tasks" section
   - Add completion date (today's date)
   - List what was implemented
   - Note any issues resolved or discoveries made
   - Update "Current Sprint" at bottom of file

2. **Commit the TASK.md changes**:
   - Use descriptive commit message like "docs: mark Task X as completed"

3. **Create Pull Request**:
   - Include comprehensive summary of implementation
   - Reference the issue number with "Closes #X"

4. **Merge Pull Request**:
   - Use merge or squash based on project preference
   - Delete branch after merge

5. **Verify Issue Closure**:
   - GitHub should auto-close the issue from PR
   - If not, manually close with reference to PR

## Important Notes:
- NEVER skip the TASK.md update - it's our primary task tracking document
- The PR should include BOTH implementation AND TASK.md update
- Use "Closes #X" in PR description for automatic issue closure