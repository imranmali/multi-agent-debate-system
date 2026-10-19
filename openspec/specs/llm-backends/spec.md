# llm-backends Specification

## Purpose
Decouples the debate engine from any model provider and supplies an offline backend so debates run deterministically without network access.

## Requirements

### Requirement: Provider-agnostic backend
The debate engine SHALL call models only through a backend interface that takes a request with a role, system prompt, user prompt and structured metadata and returns text.

#### Scenario: Custom backend
- **WHEN** a user supplies any object that implements the backend interface
- **THEN** a debate runs using only that object for model calls

### Requirement: Deterministic offline backend
The system SHALL provide an offline backend, used by default, that simulates proposer diversity, revision toward well-scored answers, critic scoring and voting, and that returns identical results for identical seeds.

#### Scenario: Same seed, same result
- **WHEN** two debates run with the offline backend and seed 7
- **THEN** both results are identical

#### Scenario: Different seeds can differ
- **WHEN** debates run with several different seeds
- **THEN** at least two of them produce different transcripts

### Requirement: Anthropic backend
The system SHALL provide a backend for Anthropic Claude models that defaults to model `claude-opus-5-5`, allows the model to be overridden, returns the concatenated text of the reply, and raises a backend error when the model refuses.

#### Scenario: Refusal surfaces as error
- **WHEN** the model's reply has stop reason `refusal`
- **THEN** the backend raises a backend error

#### Scenario: Optional dependency missing
- **WHEN** the Anthropic backend is created without the `anthropic` package installed
- **THEN** it raises a backend error explaining how to install the extra

### Requirement: Backend factory
The system SHALL create backends by name (`mock`, `anthropic`) and SHALL reject unknown names with an error listing the valid names.

#### Scenario: Unknown backend
- **WHEN** a backend named "gpt" is requested
- **THEN** creation fails with an error that lists `mock` and `anthropic`
