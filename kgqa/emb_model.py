"""Base Knowledge Graph embedding model."""
from abc import ABC, abstractmethod
import torch
from torch import nn

class KGModel(nn.Module, ABC):
    """Base Knowledge Graph Embedding model class."""

    def __init__(self, args):
        """Initialize KGModel."""
        print('****************************************')
        super(KGModel, self).__init__()
        self.rank = args.dim
        self.freeze = args.freeze
        self.dtype = args.dtype
        self.data_type = torch.double if self.dtype == 'double' else torch.float
        self.entity = nn.Embedding(args.sizes[0], self.rank, dtype=self.data_type)
        self.rel = nn.Embedding(args.sizes[1], self.rank, dtype=self.data_type)
        if self.freeze:
            self.rel = self.rel.requires_grad_(False)
            self.entity = self.entity.requires_grad_(False)

    def get_rhs(self):
        return self.entity.weight

    def get_query(self, head):
        head_e = self.entity(head)
        if len(head_e.shape) == 1:
            head_e = head_e.unsqueeze(0)

        return head_e

    def get_embeddings(self, head, question):
        return self.get_query(head), question

    @abstractmethod
    def get_queries(self, head, question):
        """Compute embedding and biases of queries.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
             lhs_e: torch.Tensor with queries' embeddings (embedding of head entities and relations)
             lhs_biases: torch.Tensor with head entities' biases
        """
        pass

    @abstractmethod
    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space.

        Args:
            lhs_e: torch.Tensor with queries' embeddings
            rhs_e: torch.Tensor with targets' embeddings
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            scores: torch.Tensor with similarity scores of queries against targets
        """
        pass


class ComplEx(KGModel):
    """Complex Knowledge Graph Embedding models.

    Attributes:
        embeddings: complex embeddings for entities and relations
    """
    def __init__(self, args):
        """Initialize a Complex KGModel."""
        super(KGModel, self).__init__(args)
        self.rank = args.rank // 2
        if self.freeze:
            self.embeddings = nn.ModuleList([nn.Embedding(args.sizes[0], 2 * self.rank, sparse=True, dtype=self.data_type).requires_grad_(False),
                                             nn.Embedding(args.sizes[1], 2 * self.rank, sparse=True, dtype=self.data_type).requires_grad_(False)])
        else:
            self.embeddings = nn.ModuleList([nn.Embedding(args.sizes[0], 2 * self.rank, sparse=True, dtype=self.data_type),
                                             nn.Embedding(args.sizes[1], 2 * self.rank, sparse=True, dtype=self.data_type).requires_grad_(False)])


    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e = lhs_e[:, :self.rank], lhs_e[:, self.rank:]
        rhs_e = rhs_e[:, :self.rank], rhs_e[:, self.rank:]
        score = lhs_e[0] @ rhs_e[0].transpose(0, 1) + lhs_e[1] @ rhs_e[1].transpose(0, 1)
        return score

    def get_embeddings(self, head, question):
        head_e = self.embeddings[0](head)
        if len(head_e.shape) == 1:
            head_e = head_e.unsqueeze(0)
        head_e = head_e[:, :self.rank], head_e[:, self.rank:]
        rel_e = question[:, :self.rank], question[:, self.rank:]

        return head_e, rel_e

    def get_rhs(self):
        return self.embeddings[0].weight

    def get_queries(self, head, question):
        """Compute embedding and biases of queries."""
        head_e, rel_e = self.get_embeddings(head, question)
        lhs_e = torch.cat([
            head_e[0] * rel_e[0] - head_e[1] * rel_e[1],
            head_e[0] * rel_e[1] + head_e[1] * rel_e[0]
        ], 1)

        return lhs_e


class DistMult(KGModel):
    """Compositional Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dot product)
    """

    def __init__(self, args):
        print('###################################################')
        super(KGModel, self).__init__()
        self.half_dim = None
        self.half = False

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.half_dim:
            rhs_e = rhs_e[:, self.half_dim:]

        score = lhs_e @ rhs_e.transpose(0, 1)

        if self.half:
            score = score * 0.5
        return score

    def get_queries(self, head, question):
        """Compute embedding and biases of queries."""
        head_e, rel_e = self.get_embeddings(head, question)
        lhs_e = head_e * rel_e
        return lhs_e


class TransE(KGModel):
    """Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args):
        super(KGModel, self).__init__()

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        score = - euc_sqdistance(lhs_e, rhs_e, eval_mode=True)
        return score

    def get_queries(self, head, question):
        head_e, rel_e = self.get_embeddings(head, question)
        lhs_e = head_e + rel_e
        return lhs_e



