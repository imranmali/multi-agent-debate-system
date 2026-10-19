# solution-critique Specification

## Purpose
Lets a critic agent evaluate every proposal against a fixed rubric so that voting, revision and confidence can use an independent quality signal.

## Requirements

### Requirement: Rubric scoring
The critic SHALL score each proposal on correctness, completeness, clarity and feasibility from 0 to 10 and SHALL report an overall score in [0, 1] equal to the mean of the four scores divided by 10.

#### Scenario: Overall score
- **WHEN** the critic scores a proposal 8, 6, 10 and 4
- **THEN** the critique's overall score is 0.7

#### Scenario: Out-of-range scores are clamped
- **WHEN** the critic returns a score of 14 or -2
- **THEN** that score is recorded as 10 or 0 respectively

### Requirement: Written feedback
Each critique SHALL include written feedback that is passed to the proposer in the next revision round.

#### Scenario: Feedback reaches the reviser
- **WHEN** a proposal receives feedback in round 1
- **THEN** the reviser's round 2 request includes that feedback

### Requirement: Concurrent critique
The system SHALL critique all proposals of a round concurrently, one critique per proposal.

#### Scenario: One critique per proposal
- **WHEN** a round has 3 proposals
- **THEN** the round has exactly 3 critiques, each referencing one proposal id

### Requirement: Degraded critiques are flagged
When the critic's reply cannot be parsed into scores, the system SHALL assign neutral scores of 5 and mark the critique as not parsed instead of failing.

#### Scenario: Unparseable critic reply
- **WHEN** the critic replies with free text that contains no scores
- **THEN** the critique has overall score 0.5 and is marked not parsed
