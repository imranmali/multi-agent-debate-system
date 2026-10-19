# consensus-voting Specification

## Purpose
Turns agents' ballots into a ranked set of positions and decides whether the swarm has reached consensus on one of them.

## Requirements

### Requirement: Answer positions
The system SHALL group proposals whose answers are equal after normalization (case-insensitive, collapsed whitespace, trailing punctuation removed) into a single position so agreeing agents do not split the vote.

#### Scenario: Equivalent answers merge
- **WHEN** two agents answer "Use Postgres." and "use postgres"
- **THEN** both proposals belong to the same position

### Requirement: Ranked ballots
Every surviving agent SHALL cast a ballot ranking the round's proposals best first; by default an agent's own proposal SHALL be excluded from its ballot, and ids missing from a ballot SHALL be appended in critic-score order.

#### Scenario: Self-vote excluded
- **WHEN** self-voting is disabled and agent A ranks its own proposal first
- **THEN** agent A's ballot does not contain its own proposal

### Requirement: Voting strategies
The system SHALL support `plurality` (one point per first choice), `borda` (n-1-rank points per ballot) and `weighted` (Borda points multiplied by the voter's own critic score, minimum weight 0.1), with `weighted` as the default.

#### Scenario: Plurality winner
- **WHEN** 3 of 5 ballots rank position X first under plurality
- **THEN** position X wins

#### Scenario: Borda rewards broad support
- **WHEN** position Y is ranked second on every ballot and position X is first on two and last on three
- **THEN** under borda position Y can outrank position X

### Requirement: Deterministic tie-break
Ties between positions SHALL break on higher mean critic score, then on the earliest proposal id.

#### Scenario: Tie on points
- **WHEN** two positions have equal points and different critic scores
- **THEN** the position with the higher critic score wins

### Requirement: Consensus rule
Support SHALL be the (weighted for `weighted`) fraction of ballots whose first choice is the winning position, and consensus SHALL be reached when support is greater than or equal to the configured threshold.

#### Scenario: Threshold met
- **WHEN** support is 0.6 and the threshold is 0.6
- **THEN** consensus is reached

#### Scenario: Threshold missed
- **WHEN** support is 0.5 and the threshold is 0.6
- **THEN** consensus is not reached
