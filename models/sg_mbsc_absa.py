import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ssegcn import GCN
from models.ssegcn_bert import GCNBert


class SGMBSCHead(nn.Module):
    """Shared/general branch plus sentiment-specific expert branches."""

    def __init__(self, input_dim, opt):
        super().__init__()
        self.opt = opt
        self.input_dim = input_dim
        self.expert_dim = opt.sg_expert_dim if opt.sg_expert_dim > 0 else input_dim
        self.temperature = opt.sg_temperature
        self.cl_weight = opt.sg_cl_weight
        self.branch_weight = getattr(opt, "sg_branch_weight", 0.0)
        self.base_weight = opt.sg_base_weight
        self.shared_ce_weight = getattr(opt, "sg_shared_ce_weight", 0.5)
        self.expert_names = ("pos", "neu", "neg")
        self.branch_names = self.expert_names + ("shared",)

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
        self.dropout = nn.Dropout(opt.sg_dropout)
        self.branch_classifiers = nn.ModuleDict({
            name: nn.Linear(self.expert_dim, 1)
            for name in self.expert_names
        })
        self.last_aux_metrics = {}
        self.expert_classifier = nn.Linear(self.expert_dim * 3, opt.polarities_dim)
        self.shared_classifier = nn.Linear(self.expert_dim, opt.polarities_dim)
        self.base_classifier = nn.Linear(input_dim, opt.polarities_dim)
        self.class_prototypes = nn.Parameter(torch.empty(len(self.expert_names), self.expert_dim))
        nn.init.xavier_uniform_(self.class_prototypes)

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

        shared_representation = pooled["shared"]
        gated = {}
        for branch in self.expert_names:
            gate = torch.sigmoid(self.gates[branch](torch.cat([pooled[branch], shared_representation], dim=-1)))
            gated[branch] = gate * pooled[branch] + (1.0 - gate) * shared_representation

        expert_fusion = torch.cat([gated["pos"], gated["neu"], gated["neg"]], dim=-1)
        expert_logits = self.expert_classifier(self.dropout(expert_fusion))
        shared_logits = self.shared_classifier(self.dropout(shared_representation))
        logits = shared_logits + expert_logits
        if base_representation is not None and self.base_weight > 0:
            logits = logits + self.base_weight * self.base_classifier(base_representation)
        if labels is not None:
            aux_loss = self._auxiliary_loss(pooled, gated, shared_logits, labels)
        else:
            aux_loss = None
            self.last_aux_metrics = {}
        return logits, aux_loss

    def _auxiliary_loss(self, pooled, gated, shared_logits, labels):
        shared_loss = self.shared_ce_weight * F.cross_entropy(shared_logits, labels)
        loss = shared_loss
        metrics = {
            "shared_ce": shared_loss.detach().item(),
        }
        if self.cl_weight > 0:
            contrastive_loss = self._contrastive_loss(gated, labels)
            loss = loss + contrastive_loss
            metrics["contrastive"] = contrastive_loss.detach().item()
        if self.branch_weight > 0:
            branch_loss = self._branch_supervision_loss(pooled, labels)
            loss = loss + branch_loss
            metrics["branch_ce"] = branch_loss.detach().item()
        metrics["aux_total"] = loss.detach().item()
        self.last_aux_metrics = metrics
        return loss

    def _label_to_branch_targets(self, labels):
        # Dataset labels: positive=0, negative=1, neutral=2.
        # Branch order in this head: pos=0, neu=1, neg=2.
        label_to_branch = torch.tensor([0, 2, 1], dtype=torch.long, device=labels.device)
        return label_to_branch[labels]

    def _branch_supervision_loss(self, pooled, labels):
        branch_logits = torch.cat([
            self.branch_classifiers[name](self.dropout(pooled[name]))
            for name in self.expert_names
        ], dim=-1)
        branch_targets = self._label_to_branch_targets(labels)
        return self.branch_weight * F.cross_entropy(branch_logits, branch_targets)

    def _contrastive_loss(self, gated, labels):
        branch_vectors = torch.stack([gated["pos"], gated["neu"], gated["neg"]], dim=1)
        branch_vectors = F.normalize(branch_vectors, p=2, dim=-1)

        branch_targets = self._label_to_branch_targets(labels)
        selected_experts = branch_vectors[torch.arange(labels.size(0), device=labels.device), branch_targets]
        prototypes = F.normalize(self.class_prototypes, p=2, dim=-1)
        similarities = torch.matmul(selected_experts, prototypes.transpose(0, 1)) / self.temperature
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
