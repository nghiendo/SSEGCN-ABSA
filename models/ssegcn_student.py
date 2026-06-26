import torch
import torch.nn as nn
import torch.nn.functional as F


class SSEGCNStudentClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt
        self.encoder = TinySSEGCNEncoder(embedding_matrix, opt)
        self.classifier = nn.Linear(opt.student_hidden_dim * 4, opt.polarities_dim)
        self.distill_proj = nn.Linear(opt.student_hidden_dim * 4, opt.teacher_feature_dim)

    def encode(self, inputs):
        return self.encoder(inputs)

    def project_for_distill(self, features):
        return self.distill_proj(features)

    def forward(self, inputs):
        features = self.encode(inputs)
        logits = self.classifier(features)
        return logits, None


class TinySSEGCNEncoder(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt
        self.word_emb = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=opt.student_freeze_word_emb,
        )
        self.pos_emb = nn.Embedding(opt.pos_size, opt.student_pos_dim, padding_idx=0)
        self.post_emb = nn.Embedding(opt.post_size, opt.student_post_dim, padding_idx=0)

        input_dim = opt.embed_dim + opt.student_pos_dim + opt.student_post_dim
        self.input_dropout = nn.Dropout(opt.student_input_dropout)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=opt.student_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        feature_dim = opt.student_hidden_dim * 2
        self.syntax_gate = nn.Linear(5, 1)
        self.token_score = nn.Linear(feature_dim, feature_dim)
        self.aspect_score = nn.Linear(feature_dim, feature_dim, bias=False)
        self.output_dropout = nn.Dropout(opt.student_output_dropout)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, lengths, short_mask = inputs
        del asp, head, deprel

        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths, dtype=torch.long, device=tok.device)
        else:
            lengths = lengths.to(tok.device).long()

        maxlen = int(lengths.max().item())
        tok = tok[:, :maxlen]
        pos = pos[:, :maxlen]
        post = post[:, :maxlen]
        mask = mask[:, :maxlen].float()
        short_mask = short_mask[:, :, :maxlen, :maxlen]

        word_emb = self.word_emb(tok)
        pos_emb = self.pos_emb(pos)
        post_emb = self.post_emb(post)
        x = torch.cat([word_emb, pos_emb, post_emb], dim=-1)
        x = self.input_dropout(x)

        # Keep the LSTM weights contiguous to avoid repeated cuDNN repacking overhead.
        self.encoder.flatten_parameters()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)

        aspect_denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        aspect_repr = (encoded * mask.unsqueeze(-1)).sum(dim=1) / aspect_denominator

        token_scores = self.token_score(encoded)
        aspect_scores = self.aspect_score(aspect_repr).unsqueeze(1)
        semantic_scores = (token_scores * aspect_scores).sum(dim=-1) / (encoded.size(-1) ** 0.5)

        syntax_prior = (short_mask == 0).float().permute(0, 2, 3, 1)
        syntax_scores = self.syntax_gate(syntax_prior).squeeze(-1)
        aspect_positions = mask.unsqueeze(1)
        syntax_scores = (syntax_scores * aspect_positions).sum(dim=-1) / aspect_denominator

        pad_mask = tok.ne(0)
        attention_scores = semantic_scores + syntax_scores
        attention_scores = attention_scores.masked_fill(~pad_mask, -1e9)
        attention = F.softmax(attention_scores, dim=-1)

        context_repr = torch.bmm(attention.unsqueeze(1), encoded).squeeze(1)
        features = torch.cat([aspect_repr, context_repr], dim=-1)
        features = self.output_dropout(features)
        return features
