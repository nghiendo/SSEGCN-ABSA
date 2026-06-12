import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ssegcn import GCN
from models.ssegcn_bert import GCNBert


class SGMBSCHead(nn.Module):
    """Shared-group multi-branch sentiment head with optional InfoNCE loss."""

    def __init__(self, input_dim, opt):
        super().__init__()
        self.opt = opt
        self.input_dim = input_dim
        self.expert_dim = opt.sg_expert_dim if opt.sg_expert_dim > 0 else input_dim
        self.temperature = opt.sg_temperature
        self.cl_weight = opt.sg_cl_weight
        self.base_weight = opt.sg_base_weight
        self.branch_names = ("pos", "neu", "neg", "shared")

        self.projections = nn.ModuleDict({
            name: nn.Linear(input_dim, self.expert_dim)
            for name in self.branch_names
        })
        self.attentions = nn.ModuleDict({
            name: nn.Linear(self.expert_dim, 1)
            for name in self.branch_names
        })
        self.gates = nn.ModuleDict({
            name: nn.Linear(self.expert_dim * 2, self.expert_dim)
            for name in ("pos", "neu", "neg")
        })
        self.shared_feedback = nn.Linear(self.expert_dim * 3, self.expert_dim)
        self.dropout = nn.Dropout(opt.sg_dropout)
        self.classifier = nn.Linear(self.expert_dim * 4, opt.polarities_dim)
        self.base_classifier = nn.Linear(input_dim, opt.polarities_dim)

    def _masked_attention_pool(self, hidden, mask, branch):
        scores = self.attentions[branch](hidden).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(weights * hidden, dim=1)

    def forward(self, sequence_outputs, sequence_mask=None, base_representation=None, labels=None):
        if sequence_mask is None:
            sequence_mask = sequence_outputs.new_ones(sequence_outputs.size()[:2])
        sequence_mask = sequence_mask[:, :sequence_outputs.size(1)].float()

        pooled = {}
        for branch in self.branch_names:
            branch_hidden = F.relu(self.projections[branch](sequence_outputs))
            pooled[branch] = self._masked_attention_pool(branch_hidden, sequence_mask, branch)

        shared_context = pooled["shared"]
        gated = {}
        for branch in ("pos", "neu", "neg"):
            gate = torch.sigmoid(self.gates[branch](torch.cat([pooled[branch], shared_context], dim=-1)))
            gated[branch] = gate * pooled[branch] + (1.0 - gate) * shared_context

        expert_feedback = torch.cat([pooled["pos"], pooled["neu"], pooled["neg"]], dim=-1)
        gated["shared"] = shared_context + torch.tanh(self.shared_feedback(expert_feedback))

        fusion = torch.cat([gated["pos"], gated["neu"], gated["neg"], gated["shared"]], dim=-1)
        logits = self.classifier(self.dropout(fusion))
        if base_representation is not None and self.base_weight > 0:
            logits = logits + self.base_weight * self.base_classifier(base_representation)
        cl_loss = self._contrastive_loss(gated, labels) if labels is not None else None
        return logits, cl_loss

    def _contrastive_loss(self, gated, labels):
        branch_vectors = torch.stack([gated["pos"], gated["neu"], gated["neg"]], dim=1)
        anchor = F.normalize(gated["shared"], p=2, dim=-1).unsqueeze(1)
        branch_vectors = F.normalize(branch_vectors, p=2, dim=-1)
        similarities = torch.sum(anchor * branch_vectors, dim=-1) / self.temperature

        # Dataset labels: positive=0, negative=1, neutral=2.
        # Branch order above: positive=0, neutral=1, negative=2.
        label_to_branch = torch.tensor([0, 2, 1], dtype=torch.long, device=labels.device)
        branch_targets = label_to_branch[labels]
        return self.cl_weight * F.cross_entropy(similarities, branch_targets)


def masked_mean(sequence_outputs, mask):
    mask = mask[:, :sequence_outputs.size(1)].float()
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return torch.sum(sequence_outputs * mask.unsqueeze(-1), dim=1) / denom


class SGMBSCClassifier(nn.Module):
    supports_contrastive = True

    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt
        embeddings = self._build_embeddings(embedding_matrix, opt)
        self.gcn = GCN(opt, embeddings, opt.hidden_dim, opt.num_layers)
        self.sg_head = SGMBSCHead(opt.hidden_dim, opt)

    def _build_embeddings(self, embedding_matrix, opt):
        word_emb = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0) if opt.pos_dim > 0 else None
        post_emb = nn.Embedding(opt.post_size, opt.post_dim, padding_idx=0) if opt.post_dim > 0 else None
        return word_emb, pos_emb, post_emb

    def forward(self, inputs, labels=None):
        tok, asp, pos, head, deprel, post, mask, length, short_mask = inputs
        sequence_outputs = self.gcn(inputs)
        sequence_mask = (tok != 0).float()
        base_representation = masked_mean(sequence_outputs, mask)
        return self.sg_head(sequence_outputs, sequence_mask, base_representation, labels)


class SGMBSCBertClassifier(nn.Module):
    supports_contrastive = True

    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)
        self.sg_head = SGMBSCHead(100, opt)

    def forward(self, inputs, labels=None):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, src_mask, aspect_mask, short_mask = inputs
        sequence_outputs = self.gcn(inputs)
        base_representation = masked_mean(sequence_outputs, aspect_mask)
        return self.sg_head(sequence_outputs, src_mask, base_representation, labels)
