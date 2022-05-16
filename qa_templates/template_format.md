## Overview

There are 4 groups of question templates: 1-hop questions, 2-hop questions, 2-hop dummy questions, advanced questions.

Each template starts reasoning from a head entity and reasons along the main chain (with entities and relations in between). The number of hops refers to the number of relations (edges) in the template's main chain. For example, 1-hop indicates that the main chain has the form: entity1-relation-entity2. 2-hop dummy templates have the form: entity1-relation1-dummy_entity-relation2-entity2, where a dummy entity is one that has no meaningful content, but connects entities multi-way.

In 1-hop and advanced questions, we included constraints. 

(1) There is at most 1 constraint in the 1-hop templates. 

(2) In advanced quesitons, there are more than 1 constraints or more complex entity-relation forms involved. 

(3) There are no constraints in 2-hop and 2-hop dummy questions. 

## Definitions of template components

### Main chain



## 1-hop questions

#### Structure: entity1-relation-entity2

## 2-hop questions

#### Structure: entity1-relation1-entity2-relation2-entity3

## 2-hop dummy questions

#### Structure: entity1-relation1-dummy-entity=relation2-entity2

## Advanced questions

#### Structure: Multi-hop, multi constraints, multi-entity





