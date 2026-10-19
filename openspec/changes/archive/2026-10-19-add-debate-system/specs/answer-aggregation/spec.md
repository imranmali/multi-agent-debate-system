# Spec Delta

## Purpose

Synthesizes one final answer from the last round of the debate and attaches an explainable confidence score to it.

## ADDED Requirements

### Requirement: Final synthesis
After the last round, an aggregator agent SHALL synthesize the final answer from the final round's proposals, critiques and the winning position; if the aggregator fails or returns no answer, the system SHALL use the winning proposal's answer.

#### Scenario: Aggregator answer used
- **WHEN** the aggregator returns a final answer
- **THEN** the result's final answer is that text and names the winning proposal

#### Scenario: Aggregator fallback
- **WHEN** the aggregator call fails
- **THEN** the result's final answer is the winning proposal's answer and the debate still succeeds

### Requirement: Confidence score
The system SHALL report a confidence in [0, 1] computed as a weighted sum of vote support, winner critic quality, answer agreement and stability, with default weights 0.4, 0.3, 0.2 and 0.1.

#### Scenario: Unanimous high-quality answer
- **WHEN** every agent agrees in every round, support is 1.0 and the winner's critic score is 1.0
- **THEN** confidence is 1.0

#### Scenario: Split debate
- **WHEN** agents are split evenly and the winner changed between rounds
- **THEN** confidence is lower than for a unanimous debate with the same critic score

### Requirement: Confidence breakdown
The result SHALL include each confidence component's value and weight so the score can be audited, and custom weights SHALL be normalized to sum to 1.

#### Scenario: Custom weights normalized
- **WHEN** weights 2, 1, 1 and 0 are configured
- **THEN** the effective weights are 0.5, 0.25, 0.25 and 0
