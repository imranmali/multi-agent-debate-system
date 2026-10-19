# Spec Delta

## Purpose

Runs a swarm of proposer agents through multiple debate rounds so that their answers can be critiqued, revised and voted on before a final answer is produced.

## ADDED Requirements

### Requirement: Concurrent proposals
The system SHALL ask every configured proposer agent for an independent answer in the first round, running those requests concurrently, and each proposer SHALL be given a distinct persona.

#### Scenario: Every agent proposes in round one
- **WHEN** a debate runs with 4 agents
- **THEN** round 1 of the transcript contains one proposal per agent, each tagged with its agent id and persona

#### Scenario: Requests run in parallel
- **WHEN** every backend call takes 100 ms and 4 agents propose
- **THEN** the proposal phase completes in well under 400 ms

### Requirement: Revision rounds
In every round after the first, the system SHALL let each agent revise its answer after seeing all peer answers and their critic scores from the previous round.

#### Scenario: Revisions see peers
- **WHEN** round 2 starts
- **THEN** each agent receives the previous round's answers and critic scores of all agents, including its own

### Requirement: Early stop on consensus
The system SHALL stop the debate after the first round in which consensus is reached and SHALL otherwise stop after the configured maximum number of rounds.

#### Scenario: Consensus in round one
- **WHEN** all agents give the same answer in round 1 with a 0.6 threshold
- **THEN** the debate runs exactly one round and reports consensus reached

#### Scenario: No consensus
- **WHEN** agents never agree and the maximum is 3 rounds
- **THEN** the debate runs 3 rounds and reports consensus not reached

### Requirement: Agent fault tolerance
The system SHALL record an error for any agent call that fails or exceeds the configured timeout and SHALL continue the debate with the remaining agents; it SHALL fail with a debate error only when no proposal survives a round.

#### Scenario: One proposer fails
- **WHEN** one of 3 proposers raises an error in round 1
- **THEN** the round contains 2 proposals and 1 recorded agent error, and the debate completes

#### Scenario: All proposers fail
- **WHEN** every proposer fails in a round
- **THEN** the debate fails with a debate error

### Requirement: Debate transcript
The system SHALL return a result containing the question, every round's proposals, critiques, votes and errors, the number of rounds run, whether consensus was reached, the final answer and the confidence, serializable to JSON.

#### Scenario: Result serializes
- **WHEN** a debate completes
- **THEN** its result converts to a JSON document containing all of those fields

### Requirement: Configuration validation
The system SHALL reject configurations with fewer than 2 agents, fewer than 1 round, a consensus threshold outside (0, 1], a non-positive timeout or an unknown voting strategy.

#### Scenario: Too few agents
- **WHEN** a debate is configured with 1 agent
- **THEN** configuration fails with a validation error naming the field
