# Overview

In total, we have 4 groups of question templates: 1-hop questions, 2-hop questions, 2-hop dummy questions, advanced questions.

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

Example: "job" is a dummy entity. Its values have the form "job@abcde12345", with "job@" followed by a 10-digit id and carries no real meanings. its entity-relatiosn include: person-has_job-job, job-job_title-job_title. where has_job and job_title are relations, job and job_title are entities.

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

Format: 

entity1-relation-entity2 [value0, value1, value2]

where entity2 needs to be equal to one of the values

### temporal_constraint

1-hop constraint, tail entity is a date, must be in a certain range

Format: 

entity1-relation-entity2 [before yyyy/mm]

entity1-relation-entity2 [after yyyy/mm]

entity1-relation-entity2 [between yyyy/mm, yyyy/mm]

### simple_constraint_nominal_2hop

1-hop constraint, tail entity must be equal to some value

Format:

entity1-relation1-entity2-relation2-entity3 [value0, value1, value2] 

where entity3 needs to be equal to one of the values

### numeric_constraint

Constraint for range selection.

Example:

"group_by" :[column1, column2] (by default group by the head entity)

"numeric": ["column", ">", 2]

### max_constraint

1-hop, tail entity is the value that should be a gwoup-wise maximum

"group_by": [column1, column2] (by default group by the head entity)

"max": "column"

### Date selection

## Ro do: can create a variable specifying the granularity (year or month)








