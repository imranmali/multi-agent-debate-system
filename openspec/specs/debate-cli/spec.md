# debate-cli Specification

## Purpose
Gives users a command-line entry point to run a debate and read the result as a human-readable report or as JSON.

## Requirements

### Requirement: Run a debate from the command line
The `debate` command SHALL take a question and options for agent count, rounds, consensus threshold, voting strategy, backend, model and seed, and SHALL run a debate with them.

#### Scenario: Default run
- **WHEN** a user runs `debate "Which database should we use?"`
- **THEN** a debate runs with the offline backend and the command exits with status 0

### Requirement: Output formats
The command SHALL print a human-readable report with the final answer, confidence and per-round summary by default, and SHALL print the full result as JSON when `--json` is given.

#### Scenario: JSON output
- **WHEN** a user runs the command with `--json`
- **THEN** standard output is a single valid JSON document containing the final answer and confidence

### Requirement: Exit codes
The command SHALL exit with status 2 for invalid arguments or configuration and status 1 when the debate or backend fails, printing the error to standard error.

#### Scenario: Invalid agent count
- **WHEN** a user runs the command with `--agents 1`
- **THEN** it exits with status 2 and prints the validation error to standard error

#### Scenario: Backend failure
- **WHEN** the selected backend cannot be created
- **THEN** the command exits with status 1
