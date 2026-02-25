# CLAUDE.md

This file provides guidance for AI assistants working with the **sena3fx** repository.

## Repository Overview

**sena3fx** is a newly initialized project. The codebase is in its early stages — contribute carefully and establish good patterns from the start.

- **Primary branch**: `master`
- **Remote**: GitHub (via `origin`)

## Project Structure

```
sena3fx/
├── CLAUDE.md        # AI assistant guidance (this file)
├── README.md        # Project overview
└── .git/            # Git version control
```

> Update this section as the project grows with source directories, configs, and tooling.

## Development Workflow

### Branching

- Work on feature branches; never push directly to `master` without review.
- Branch naming: `<scope>/<short-description>` (e.g., `feat/add-auth`, `fix/null-check`).

### Commits

- Write clear, imperative-mood commit messages (e.g., "Add user login endpoint").
- Keep commits focused — one logical change per commit.
- Do not amend published commits.

### Pull Requests

- Provide a summary and test plan in PR descriptions.
- Ensure CI passes (once configured) before merging.

## Code Conventions

### General

- Prefer simplicity and readability over cleverness.
- Do not add features, abstractions, or error handling beyond what is needed for the current task.
- Delete unused code rather than commenting it out.

### Security

- Never commit secrets, credentials, or `.env` files.
- Validate all external input at system boundaries.
- Follow OWASP best practices.

## Testing

> No test framework has been configured yet. When one is added, document it here:
> - Test runner and command (e.g., `npm test`, `mvn test`, `pytest`)
> - Test file location conventions
> - Minimum coverage expectations

## Build & Run

> No build system has been configured yet. When one is added, document it here:
> - Build command (e.g., `npm run build`, `mvn package`)
> - Run command (e.g., `npm start`, `java -jar ...`)
> - Required environment variables

## CI/CD

> No CI/CD pipeline has been configured yet. When one is added, document it here:
> - Pipeline location (e.g., `.github/workflows/`)
> - Required checks before merge

## Dependencies

> No dependency management has been configured yet. When one is added, document it here:
> - Package manager (e.g., npm, maven, pip)
> - Lock file policy
> - How to add/update dependencies

## Notes for AI Assistants

- Read existing code before modifying it.
- Do not create files unless strictly necessary — prefer editing existing ones.
- Keep changes minimal and scoped to the task at hand.
- When the project structure evolves, update this CLAUDE.md to reflect the current state.
