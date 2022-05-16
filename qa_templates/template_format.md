# Overview

There are 4 groups of question templates: 1-hop questions, 2-hop questions, 2-hop dummy questions, advanced questions.

In 1-hop and advanced questions, we included constraints. 

(1) There is at most 1 constraint in the 1-hop templates. 

(2) In advanced quesitons, there are more than 1 constraints or more complex entity-relation forms involved. 

(3) There are no constraints in 2-hop and 2-hop dummy questions. 

# Definitions of template components

## Question

The question is a question in natural language form. It has a head entity from which the reasoning starts. The head entity is marked in square brakets. 

Example: country of [org], [person] is the sponsor of which events

## Main chain

The main route of reasoning for a question template. It is in the form of entity-relation-entity. It starts with a head entity and is connected with relations and other entities.

Example: The main chain of the question "[person] is the sponsor of which events" is "person-sponsor-event", where "person" and "event" are entities and "sponsor" is the relation.

## Number of hops

The number of hops refers to the number of relations (edges) in the template's main chain. 

Example: 1-hop indicates that the main chain has the form: entity1-relation-entity2. 2-hop dummy templates have the form: entity1-relation1-dummy_entity-relation2-entity2. 

## Dummy entity

where a dummy entity is one that has no meaningful content, but connects entities multi-way.

Example: person-has_job-job, job-job_title-job_title

## 1-hop questions

#### Structure: entity1-relation-entity2

## 2-hop questions

#### Structure: entity1-relation1-entity2-relation2-entity3

## 2-hop dummy questions

#### Structure: entity1-relation1-dummy-entity=relation2-entity2

## Advanced questions

#### Structure: Multi-hop, multi constraints, multi-entity

## Constraints

### simple_constraint_nominal_1hop

1-hop constraint, tail entity must be equal to some value

### temporal_constraint

1-hop constraint, tail entity is a date, must be in a certain range

### simple_constraint_nominal_2hop

1-hop constraint, tail entity must be equal to some value

### count_and_groupby (to be deleted)

### max_constraint







