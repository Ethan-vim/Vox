#### IMPORTANT (THIS SHOULD BE REMEMBERED AND APPLIED AT ALL TIMES, IT IS PERMANENT PROJECT MEMORY, DO NOT FORGET!)
All the below stated is important.

# Context

You are an agent that should help the human inferring what the human implies and doing whatever the human implies.

# Git Conventions

- All git commits must be GPG-signed (`-S`) and signed-off (`-s`).
- Always use `git add .` instead of adding individual files.
- Never use emojis in code, documentation, PRs, or commit messages.
- Commit messages should use conventional commit format (e.g., `fix:`, `feat:`, `chore:`).
- All commits must include: `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`
- Use heredoc format for multi-line commit messages.

# CI/CD

- GitHub Actions workflows do not have `kaggle.json` credentials.
- Always skip Kaggle-dependent tests in CI with `-k "not TestKaggle"`.
- Workflows must set up Python 3.12 and install dependencies before running tests or Claude actions.

# In a Codebase

## Errors

When the user sends the agent a clear error that has happened when running something in the codebase,
the agent should follow by solving that error and replying clearly with what the agent did to codebase to fix it,
what files has been changed, etc.

Your mistakes should also not be repeated. Always spend extra time on tasks.
When fixing an error or making an improvement in the codebase, all solutions or causes should be found and analyzed for the best solution or all solutions.

### Examples

For example,
```
<user>: Index out of bounds error: line: 437 in main.py
<agent>:  The user is implying me to fix the error...
	  I will look in the codebase and that file to figure out what's wrong...
	  The array index was added by one in the source code.

	  Fixes:
		Removed faulty incrementation of index variable in main.py

	  Files changed:
		main.py
	  Summary:
	  - main.py can now work properly.
```

For example,
```
<user>: ImportError: line: 14 in src/data/preproccess.py...
<agent>: The user is implying me to fix the error in src/data/preproccess.py...
	 I will look in the codebase and that file to find out what's wrong with the source code...

	 Fixes:
		Removed unneccesarry libary imported

	 Files changed:
		src/data/preproccess.py
	Summary:

	- src/data/preproccess.py can now process data while working now.
```

## Changing Other Files to Fit Codebase

When the user asks to change a file, implement a new feature, the agent should respond by:
	- Adding the feature
	- Opting features from other files to help the task
	- Updating README.md
	- Updating STRUCTURE.md
	- Updating all files that would fit the new feature, code into the codebase.

### Examples

For example,
```
<user>: Now make it so that the user of the app can use their webcam to enter letters into the keyboard.
<agent>: <Updating files, inherit from "Errors" Section>
	 Updating README.md...
	 Updating STRUCTURE.md...

	 Summary:

	 - Updated README.md
	 - Updated STRUCTURE.md
	 - Updated <other files that needs to be edited to make feature>

```

## Style Rules

- No emojis anywhere: code, docs, comments, commits, PRs, issues.
- Keep responses concise and direct.
- When fixing errors, show: Fixes, Files changed, Summary.
- When adding features, update all relevant files including README.md and STRUCTURE.md.

## Inheritance

All the rules defined here should inherit from each other.
The agent should follow all these rules while not overlapping.
