'''
Description: 
version: 
Author: chenhao
Date: 2021-06-09 14:17:37
'''
import os
import sys
import copy
import random
import logging
import argparse
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import BertModel
try:
    from torch.optim import AdamW
except ImportError:
    from transformers import AdamW

from models.ssegcn import SSEGCNClassifier
from models.ssegcn_bert import SSEGCNBertClassifier
from models.ssegcn_student import SSEGCNStudentClassifier
from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix, Tokenizer4BertGCN, ABSAGCNData, KDABSADataset
from prepare_vocab import VocabHelp

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

INPUT_COLSES = {
    'ssegcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'short_mask'],
    'ssegcnbert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'src_mask', 'aspect_mask', 'short_mask'],
    'ssegcnbertstudent': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'short_mask'],
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
        self.best_model = None
        self.teacher_model = None
        self.student_input_cols = None
        self.teacher_input_cols = None

        if opt.model_name == 'ssegcnbertstudent':
            self._build_student_kd_pipeline()
        elif 'bert' in opt.model_name:
            tokenizer = Tokenizer4BertGCN(opt.max_length, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
            trainset = ABSAGCNData(opt.dataset_file['train'], tokenizer, opt=opt)
            testset = ABSAGCNData(opt.dataset_file['test'], tokenizer, opt=opt)
        else:    
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']], 
                max_length=opt.max_length, 
                data_file='{}/{}_tokenizer.dat'.format(opt.vocab_dir, opt.dataset))
            embedding_matrix = build_embedding_matrix(
                vocab=tokenizer.vocab, 
                embed_dim=opt.embed_dim, 
                data_file='{}/{}d_{}_embedding_matrix.dat'.format(opt.vocab_dir, str(opt.embed_dim), opt.dataset))

            logger.info("Loading vocab...")
            token_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_tok.vocab')    # token
            post_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_post.vocab')    # position
            pos_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pos.vocab')      # POS
            dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')      # deprel
            pol_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pol.vocab')      # polarity
            logger.info("token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)))

            # opt.tok_size = len(token_vocab)
            opt.post_size = len(post_vocab)
            opt.pos_size = len(pos_vocab)
            
            vocab_help = (post_vocab, pos_vocab, dep_vocab, pol_vocab)
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
            trainset = SentenceDataset(opt.dataset_file['train'], tokenizer, opt=opt, vocab_help=vocab_help)
            testset = SentenceDataset(opt.dataset_file['test'], tokenizer, opt=opt, vocab_help=vocab_help)

        if opt.model_name != 'ssegcnbertstudent':
            self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
            self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size)

        if opt.device.type == 'cuda' and torch.cuda.is_available():
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated()))
        self._print_args()

    def _load_word_side_resources(self):
        tokenizer = build_tokenizer(
            fnames=[self.opt.dataset_file['train'], self.opt.dataset_file['test']],
            max_length=self.opt.max_length,
            data_file='{}/{}_tokenizer.dat'.format(self.opt.vocab_dir, self.opt.dataset))
        embedding_matrix = build_embedding_matrix(
            vocab=tokenizer.vocab,
            embed_dim=self.opt.embed_dim,
            data_file='{}/{}d_{}_embedding_matrix.dat'.format(self.opt.vocab_dir, str(self.opt.embed_dim), self.opt.dataset))

        logger.info("Loading vocab...")
        token_vocab = VocabHelp.load_vocab(self.opt.vocab_dir + '/vocab_tok.vocab')
        post_vocab = VocabHelp.load_vocab(self.opt.vocab_dir + '/vocab_post.vocab')
        pos_vocab = VocabHelp.load_vocab(self.opt.vocab_dir + '/vocab_pos.vocab')
        dep_vocab = VocabHelp.load_vocab(self.opt.vocab_dir + '/vocab_dep.vocab')
        pol_vocab = VocabHelp.load_vocab(self.opt.vocab_dir + '/vocab_pol.vocab')
        logger.info("token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(
            len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)))

        self.opt.post_size = len(post_vocab)
        self.opt.pos_size = len(pos_vocab)
        vocab_help = (post_vocab, pos_vocab, dep_vocab, pol_vocab)
        return tokenizer, embedding_matrix, vocab_help

    def _build_student_kd_pipeline(self):
        word_tokenizer, embedding_matrix, vocab_help = self._load_word_side_resources()
        bert_tokenizer = Tokenizer4BertGCN(self.opt.max_length, self.opt.pretrained_bert_name)

        self.model = self.opt.model_class(embedding_matrix, self.opt).to(self.opt.device)

        train_sentence = SentenceDataset(self.opt.dataset_file['train'], word_tokenizer, opt=self.opt, vocab_help=vocab_help)
        test_sentence = SentenceDataset(self.opt.dataset_file['test'], word_tokenizer, opt=self.opt, vocab_help=vocab_help)
        train_bert = ABSAGCNData(self.opt.dataset_file['train'], bert_tokenizer, opt=self.opt)
        test_bert = ABSAGCNData(self.opt.dataset_file['test'], bert_tokenizer, opt=self.opt)

        trainset = KDABSADataset(train_sentence, train_bert)
        testset = KDABSADataset(test_sentence, test_bert)
        self.train_dataloader = DataLoader(dataset=trainset, batch_size=self.opt.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=self.opt.batch_size)

        self.student_input_cols = INPUT_COLSES['ssegcn']
        self.teacher_input_cols = INPUT_COLSES['ssegcnbert']
        self._load_teacher_model()

    def _load_teacher_model(self):
        teacher_path = self.opt.teacher_path
        if not teacher_path or str(teacher_path).lower() in ('auto', 'latest'):
            teacher_path = self._find_latest_teacher_checkpoint()
        if not teacher_path:
            raise ValueError(
                'No teacher checkpoint found. Train ssegcnbert first or pass --teacher_path explicitly.'
            )

        bert = BertModel.from_pretrained(self.opt.pretrained_bert_name)
        teacher = SSEGCNBertClassifier(bert, self.opt).to(self.opt.device)
        logger.info('Loading teacher checkpoint: {}'.format(teacher_path))
        state_dict = torch.load(teacher_path, map_location=self.opt.device)
        teacher.load_state_dict(state_dict, strict=True)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        self.teacher_model = teacher

    def _find_latest_teacher_checkpoint(self):
        pattern = os.path.join(
            '.', 'state_dict', 'ssegcnbert_{}_acc_*_f1_*'.format(self.opt.dataset)
        )
        candidates = glob.glob(pattern)
        if not candidates:
            return None
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]
    
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = int(torch.prod(torch.tensor(p.shape)))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')
        
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
    
    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)   # xavier_uniform_
                else:
                    stdv = 1. / (p.shape[0]**0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_bert_optimizer(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        diff_part = ["bert.embeddings", "bert.encoder"]

        if self.opt.diff_lr:
            logger.info("layered learning rate on")
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.learning_rate
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.learning_rate
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)

        else:
            logger.info("bert learning rate on")
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.opt.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer

    def _kd_instance_weights(self, teacher_logits, student_logits):
        num_classes = teacher_logits.size(-1)
        normalizer = float(np.log(num_classes)) if num_classes > 1 else 1.0

        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            teacher_entropy = -(teacher_probs * torch.log(teacher_probs.clamp_min(1e-12))).sum(dim=-1)
            teacher_confidence = 1.0 - (teacher_entropy / normalizer)

            student_probs = F.softmax(student_logits, dim=-1)
            student_entropy = -(student_probs * torch.log(student_probs.clamp_min(1e-12))).sum(dim=-1)
            student_uncertainty = student_entropy / normalizer

            sample_weights = teacher_confidence * student_uncertainty
            sample_weights = sample_weights.clamp_min(self.opt.kd_min_weight)

            if self.opt.kd_normalize_weights:
                sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-12)

        return sample_weights

    def _weighted_mean(self, values, weights):
        return (values * weights).sum() / weights.sum().clamp_min(1e-12)

    def _standardize_logits(self, logits):
        mean = logits.mean(dim=-1, keepdim=True)
        std = logits.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (logits - mean) / std

    def _prepare_kd_logits(self, logits):
        if self.opt.kd_logit_standardize:
            return self._standardize_logits(logits)
        return logits

    def _cat_target_other(self, probs, target_mask, other_mask):
        target_prob = (probs * target_mask).sum(dim=1, keepdim=True)
        other_prob = (probs * other_mask).sum(dim=1, keepdim=True)
        return torch.cat([target_prob, other_prob], dim=1)

    def _dkd_loss(self, student_logits, teacher_logits, targets, temperature, sample_weights):
        target_mask = torch.zeros_like(student_logits).scatter_(1, targets.unsqueeze(1), 1).bool()
        other_mask = ~target_mask

        student_probs = F.softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        student_two_way = self._cat_target_other(student_probs, target_mask, other_mask)
        teacher_two_way = self._cat_target_other(teacher_probs, target_mask, other_mask)
        tckd_per_sample = F.kl_div(
            torch.log(student_two_way.clamp_min(1e-12)),
            teacher_two_way,
            reduction='none',
        ).sum(dim=-1) * (temperature ** 2)

        large_negative = 1e9
        teacher_non_target = teacher_logits / temperature - target_mask.float() * large_negative
        student_non_target = student_logits / temperature - target_mask.float() * large_negative
        nckd_per_sample = F.kl_div(
            F.log_softmax(student_non_target, dim=-1),
            F.softmax(teacher_non_target, dim=-1),
            reduction='none',
        ).sum(dim=-1) * (temperature ** 2)

        if sample_weights is not None:
            tckd_loss = self._weighted_mean(tckd_per_sample, sample_weights)
            nckd_loss = self._weighted_mean(nckd_per_sample, sample_weights)
        else:
            tckd_loss = tckd_per_sample.mean()
            nckd_loss = nckd_per_sample.mean()

        total = self.opt.kd_dkd_target_weight * tckd_loss + self.opt.kd_dkd_non_target_weight * nckd_loss
        return total, tckd_loss, nckd_loss

    def _pearson_distance(self, student_values, teacher_values, dim=-1):
        student_centered = student_values - student_values.mean(dim=dim, keepdim=True)
        teacher_centered = teacher_values - teacher_values.mean(dim=dim, keepdim=True)

        student_norm = student_centered.norm(dim=dim)
        teacher_norm = teacher_centered.norm(dim=dim)
        denom = (student_norm * teacher_norm).clamp_min(1e-6)
        correlation = (student_centered * teacher_centered).sum(dim=dim) / denom
        distance = 1.0 - correlation

        degenerate = (student_norm < 1e-6) | (teacher_norm < 1e-6)
        distance = torch.where(degenerate, torch.zeros_like(distance), distance)
        return distance

    def _dist_loss(self, student_logits, teacher_logits, temperature, sample_weights):
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        inter_per_sample = self._pearson_distance(student_probs, teacher_probs, dim=-1)
        if sample_weights is not None:
            inter_loss = self._weighted_mean(inter_per_sample, sample_weights)
        else:
            inter_loss = inter_per_sample.mean()

        intra_per_class = self._pearson_distance(
            student_probs.transpose(0, 1),
            teacher_probs.transpose(0, 1),
            dim=-1,
        )
        intra_loss = intra_per_class.mean()

        total = (
            self.opt.kd_dist_inter_weight * inter_loss
            + self.opt.kd_dist_intra_weight * intra_loss
        )
        return total, inter_loss, intra_loss

    def _pairwise_distance(self, features):
        distances = torch.cdist(features, features, p=2)
        valid = distances > 0
        mean_distance = distances[valid].mean() if valid.any() else distances.new_tensor(1.0)
        return distances / mean_distance.clamp_min(1e-6)

    def _rkd_distance_loss(self, student_features, teacher_features):
        student_distance = self._pairwise_distance(student_features)
        teacher_distance = self._pairwise_distance(teacher_features)
        return F.smooth_l1_loss(student_distance, teacher_distance)

    def _rkd_angle_loss(self, student_features, teacher_features):
        student_diff = F.normalize(student_features.unsqueeze(0) - student_features.unsqueeze(1), p=2, dim=-1)
        teacher_diff = F.normalize(teacher_features.unsqueeze(0) - teacher_features.unsqueeze(1), p=2, dim=-1)
        student_angle = torch.bmm(student_diff, student_diff.transpose(1, 2))
        teacher_angle = torch.bmm(teacher_diff, teacher_diff.transpose(1, 2))
        return F.smooth_l1_loss(student_angle, teacher_angle)

    def _contrastive_kd_loss(self, student_features, teacher_features):
        student_norm = F.normalize(student_features, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
        logits = torch.matmul(student_norm, teacher_norm.transpose(0, 1)) / self.opt.kd_contrastive_temperature
        targets = torch.arange(student_features.size(0), device=student_features.device)
        loss_st = F.cross_entropy(logits, targets)
        loss_ts = F.cross_entropy(logits.transpose(0, 1), targets)
        return 0.5 * (loss_st + loss_ts)

    def _margin_kd_loss(self, student_logits, teacher_logits, targets, sample_weights):
        target_idx = targets.unsqueeze(1)
        student_target = student_logits.gather(1, target_idx).squeeze(1)
        teacher_target = teacher_logits.gather(1, target_idx).squeeze(1)

        mask = torch.zeros_like(student_logits).scatter_(1, target_idx, 1).bool()
        student_rival = student_logits.masked_fill(mask, -1e9).max(dim=1).values
        teacher_rival = teacher_logits.masked_fill(mask, -1e9).max(dim=1).values

        student_margin = student_target - student_rival
        teacher_margin = teacher_target - teacher_rival
        per_sample = F.smooth_l1_loss(student_margin, teacher_margin, reduction='none')
        if sample_weights is not None:
            return self._weighted_mean(per_sample, sample_weights)
        return per_sample.mean()

    def _rank_kd_loss(self, student_logits, teacher_logits, temperature, sample_weights):
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_scores = student_logits / max(self.opt.kd_rank_temperature, 1e-6)
        teacher_scores = teacher_logits / temperature

        teacher_diff = teacher_scores.unsqueeze(2) - teacher_scores.unsqueeze(1)
        student_diff = student_scores.unsqueeze(2) - student_scores.unsqueeze(1)
        teacher_sign = teacher_diff.sign()
        pair_weights = (teacher_probs.unsqueeze(2) - teacher_probs.unsqueeze(1)).abs().detach()

        upper_mask = torch.triu(torch.ones_like(pair_weights), diagonal=1)
        pair_mask = upper_mask * (pair_weights > 0).float()
        pair_losses = F.softplus(-teacher_sign * student_diff) * pair_weights * pair_mask

        normalizer = (pair_weights * pair_mask).sum(dim=(1, 2)).clamp_min(1e-6)
        per_sample = pair_losses.sum(dim=(1, 2)) / normalizer
        if sample_weights is not None:
            return self._weighted_mean(per_sample, sample_weights)
        return per_sample.mean()

    def _prototype_kd_loss(self, student_features, teacher_features, teacher_logits):
        teacher_probs = F.softmax(teacher_logits / self.opt.kd_temperature, dim=-1).detach()

        student_proto, teacher_proto = [], []
        for class_idx in range(teacher_probs.size(1)):
            class_weight = teacher_probs[:, class_idx:class_idx + 1]
            normalizer = class_weight.sum().clamp_min(1e-6)
            student_proto.append((student_features * class_weight).sum(dim=0) / normalizer)
            teacher_proto.append((teacher_features * class_weight).sum(dim=0) / normalizer)

        student_proto = torch.stack(student_proto, dim=0)
        teacher_proto = torch.stack(teacher_proto, dim=0)

        proto_align = F.mse_loss(student_proto, teacher_proto)
        student_rel = torch.matmul(F.normalize(student_proto, p=2, dim=-1), F.normalize(student_proto, p=2, dim=-1).transpose(0, 1))
        teacher_rel = torch.matmul(F.normalize(teacher_proto, p=2, dim=-1), F.normalize(teacher_proto, p=2, dim=-1).transpose(0, 1))
        proto_relation = F.mse_loss(student_rel, teacher_rel)
        total = (
            self.opt.kd_proto_weight * proto_align
            + self.opt.kd_proto_relation_weight * proto_relation
        )
        return total, proto_align, proto_relation

    def _get_feature_target(self, teacher_logits, teacher_features):
        if self.opt.kd_feature_mode == 'teacher_hidden':
            return teacher_features
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        return teacher_probs @ self.teacher_model.classifier.weight

    def _feature_kd_loss(self, student_features, feature_target, sample_weights):
        if self.opt.kd_feature_loss == 'cosine':
            per_sample = 1.0 - F.cosine_similarity(student_features, feature_target, dim=-1)
        else:
            per_sample = F.mse_loss(student_features, feature_target, reduction='none').mean(dim=-1)

        if sample_weights is not None:
            return self._weighted_mean(per_sample, sample_weights)
        return per_sample.mean()

    def _kd_loss(self, student_logits, student_features, teacher_logits, teacher_features, targets, temperature):
        if self.opt.kd_use_instance_weighting:
            sample_weights = self._kd_instance_weights(teacher_logits, student_logits)
        else:
            sample_weights = None

        prepared_student_logits = self._prepare_kd_logits(student_logits)
        prepared_teacher_logits = self._prepare_kd_logits(teacher_logits)

        if self.opt.kd_logit_mode == 'dkd':
            kd_logits_loss, tckd_loss, nckd_loss = self._dkd_loss(
                prepared_student_logits,
                prepared_teacher_logits,
                targets,
                temperature,
                sample_weights,
            )
            kd_dist_inter = kd_logits_loss.new_tensor(0.0)
            kd_dist_intra = kd_logits_loss.new_tensor(0.0)
        elif self.opt.kd_logit_mode == 'dist':
            kd_logits_loss, kd_dist_inter, kd_dist_intra = self._dist_loss(
                prepared_student_logits,
                prepared_teacher_logits,
                temperature,
                sample_weights,
            )
            tckd_loss = kd_logits_loss.new_tensor(0.0)
            nckd_loss = kd_logits_loss.new_tensor(0.0)
        else:
            soft_targets = F.softmax(prepared_teacher_logits / temperature, dim=-1)
            student_log_probs = F.log_softmax(prepared_student_logits / temperature, dim=-1)
            kd_logits_per_sample = F.kl_div(student_log_probs, soft_targets, reduction='none').sum(dim=-1)
            kd_logits_per_sample = kd_logits_per_sample * (temperature ** 2)
            if sample_weights is not None:
                kd_logits_loss = self._weighted_mean(kd_logits_per_sample, sample_weights)
            else:
                kd_logits_loss = kd_logits_per_sample.mean()
            tckd_loss = kd_logits_loss.new_tensor(0.0)
            nckd_loss = kd_logits_loss.new_tensor(0.0)
            kd_dist_inter = kd_logits_loss.new_tensor(0.0)
            kd_dist_intra = kd_logits_loss.new_tensor(0.0)

        feature_target = self._get_feature_target(teacher_logits, teacher_features)
        kd_feature_loss = self._feature_kd_loss(student_features, feature_target, sample_weights)

        kd_relation_distance = kd_feature_loss.new_tensor(0.0)
        kd_relation_angle = kd_feature_loss.new_tensor(0.0)
        if self.opt.kd_relation_weight > 0:
            kd_relation_distance = self._rkd_distance_loss(student_features, feature_target)
            kd_relation_angle = self._rkd_angle_loss(student_features, feature_target)

        kd_contrastive_loss = kd_feature_loss.new_tensor(0.0)
        if self.opt.kd_contrastive_weight > 0:
            kd_contrastive_loss = self._contrastive_kd_loss(student_features, feature_target)

        kd_margin_loss = kd_feature_loss.new_tensor(0.0)
        if self.opt.kd_margin_weight > 0:
            kd_margin_loss = self._margin_kd_loss(student_logits, teacher_logits, targets, sample_weights)

        kd_rank_loss = kd_feature_loss.new_tensor(0.0)
        if self.opt.kd_rank_weight > 0:
            kd_rank_loss = self._rank_kd_loss(
                prepared_student_logits,
                prepared_teacher_logits,
                temperature,
                sample_weights,
            )

        kd_proto_loss = kd_feature_loss.new_tensor(0.0)
        kd_proto_align = kd_feature_loss.new_tensor(0.0)
        kd_proto_relation = kd_feature_loss.new_tensor(0.0)
        if self.opt.kd_proto_weight > 0 or self.opt.kd_proto_relation_weight > 0:
            kd_proto_loss, kd_proto_align, kd_proto_relation = self._prototype_kd_loss(
                student_features=student_features,
                teacher_features=feature_target,
                teacher_logits=teacher_logits,
            )

        total_loss = (
            self.opt.kd_beta * kd_logits_loss
            + self.opt.kd_gamma * kd_feature_loss
            + self.opt.kd_relation_weight * (
                kd_relation_distance + self.opt.kd_relation_angle_weight * kd_relation_angle
            )
            + self.opt.kd_contrastive_weight * kd_contrastive_loss
            + self.opt.kd_margin_weight * kd_margin_loss
            + self.opt.kd_rank_weight * kd_rank_loss
            + kd_proto_loss
        )

        components = {
            'kd_logit': kd_logits_loss,
            'kd_feature': kd_feature_loss,
            'kd_tckd': tckd_loss,
            'kd_nckd': nckd_loss,
            'kd_dist_inter': kd_dist_inter,
            'kd_dist_intra': kd_dist_intra,
            'kd_relation_distance': kd_relation_distance,
            'kd_relation_angle': kd_relation_angle,
            'kd_contrastive': kd_contrastive_loss,
            'kd_margin': kd_margin_loss,
            'kd_rank': kd_rank_loss,
            'kd_proto': kd_proto_loss,
            'kd_proto_align': kd_proto_align,
            'kd_proto_relation': kd_proto_relation,
        }
        return total_loss, components, sample_weights

    def _epoch_progress(self, epoch_idx):
        if self.opt.num_epoch <= 1:
            return 1.0
        return epoch_idx / float(self.opt.num_epoch - 1)

    def _current_kd_temperature(self, epoch_idx):
        if self.opt.kd_temperature_schedule == 'constant':
            return self.opt.kd_temperature

        progress = self._epoch_progress(epoch_idx)
        start = self.opt.kd_temperature_start
        end = self.opt.kd_temperature_end
        if self.opt.kd_temperature_schedule == 'cosine':
            ratio = 0.5 * (1.0 - np.cos(np.pi * progress))
        else:
            ratio = progress
        return start + (end - start) * ratio

    def _current_kd_scale(self, epoch_idx):
        warmup_epochs = max(0, self.opt.kd_warmup_epochs)
        if epoch_idx < warmup_epochs:
            return 0.0

        ramp_epochs = max(1, self.opt.kd_ramp_epochs)
        progress = min(1.0, (epoch_idx - warmup_epochs + 1) / float(ramp_epochs))
        if self.opt.kd_scale_schedule == 'cosine':
            return 0.5 * (1.0 - np.cos(np.pi * progress))
        return progress

    def _train_kd(self, criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        best_saved_acc = 0
        best_saved_f1 = 0
        global_step = 0
        model_path = ''
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            kd_temperature = self._current_kd_temperature(epoch)
            kd_scale = self._current_kd_scale(epoch)
            hard_loss_weight = self.opt.kd_alpha + (1.0 - kd_scale) * (1.0 - self.opt.kd_alpha)
            logger.info(
                'kd curriculum: temp={:.4f}, kd_scale={:.4f}, hard_weight={:.4f}'.format(
                    kd_temperature,
                    kd_scale,
                    hard_loss_weight,
                )
            )
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                self.model.train()
                optimizer.zero_grad()

                student_inputs = [sample_batched[col].to(self.opt.device) for col in self.student_input_cols]
                teacher_inputs = [sample_batched[col].to(self.opt.device) for col in self.teacher_input_cols]
                targets = sample_batched['polarity'].to(self.opt.device)

                student_features = self.model.encode(student_inputs)
                projected_features = self.model.project_for_distill(student_features)
                student_logits = self.model.classifier(student_features)

                with torch.no_grad():
                    teacher_logits, _ = self.teacher_model(teacher_inputs)
                    teacher_features = self.teacher_model.encode(teacher_inputs)

                hard_loss = criterion(student_logits, targets)
                kd_loss, kd_components, sample_weights = self._kd_loss(
                    student_logits,
                    projected_features,
                    teacher_logits,
                    teacher_features,
                    targets,
                    kd_temperature,
                )
                loss = hard_loss_weight * hard_loss + kd_scale * kd_loss

                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(student_logits, -1) == targets).sum().item()
                    n_total += len(student_logits)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                    if f1 > max_f1:
                        max_f1 = f1
                    if (f1 > best_saved_f1) or (f1 == best_saved_f1 and test_acc > best_saved_acc):
                        best_saved_acc = test_acc
                        best_saved_f1 = f1
                        if test_acc > max_test_acc_overall:
                            os.makedirs('./state_dict', exist_ok=True)
                            model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(
                                self.opt.model_name, self.opt.dataset, test_acc, f1)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(model_path))
                    log_message = (
                        'loss: {:.4f}, hard: {:.4f}, kd_total: {:.4f}, kd_logit: {:.4f}, kd_feat: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(
                            loss.item(),
                            hard_loss.item(),
                            (kd_scale * kd_loss).item(),
                            kd_components['kd_logit'].item(),
                            kd_components['kd_feature'].item(),
                            train_acc,
                            test_acc,
                            f1,
                        )
                    )
                    log_message += ', temp: {:.2f}, kd_scale: {:.2f}, hard_w: {:.2f}'.format(
                        kd_temperature,
                        kd_scale,
                        hard_loss_weight,
                    )
                    if self.opt.kd_logit_mode == 'dkd':
                        log_message += ', tckd: {:.4f}, nckd: {:.4f}'.format(
                            kd_components['kd_tckd'].item(),
                            kd_components['kd_nckd'].item(),
                        )
                    if self.opt.kd_logit_mode == 'dist':
                        log_message += ', dist_inter: {:.4f}, dist_intra: {:.4f}'.format(
                            kd_components['kd_dist_inter'].item(),
                            kd_components['kd_dist_intra'].item(),
                        )
                    if self.opt.kd_relation_weight > 0:
                        log_message += ', rkd_d: {:.4f}, rkd_a: {:.4f}'.format(
                            kd_components['kd_relation_distance'].item(),
                            kd_components['kd_relation_angle'].item(),
                        )
                    if self.opt.kd_contrastive_weight > 0:
                        log_message += ', kd_ctr: {:.4f}'.format(kd_components['kd_contrastive'].item())
                    if self.opt.kd_margin_weight > 0:
                        log_message += ', kd_margin: {:.4f}'.format(kd_components['kd_margin'].item())
                    if self.opt.kd_rank_weight > 0:
                        log_message += ', kd_rank: {:.4f}'.format(kd_components['kd_rank'].item())
                    if self.opt.kd_proto_weight > 0 or self.opt.kd_proto_relation_weight > 0:
                        log_message += ', kd_proto: {:.4f}, kd_proto_rel: {:.4f}'.format(
                            kd_components['kd_proto_align'].item(),
                            kd_components['kd_proto_relation'].item(),
                        )
                    if sample_weights is not None:
                        log_message += ', kd_w_mean: {:.4f}, kd_w_min: {:.4f}, kd_w_max: {:.4f}'.format(
                            sample_weights.mean().item(),
                            sample_weights.min().item(),
                            sample_weights.max().item(),
                        )
                    logger.info(log_message)
        return max_test_acc, max_f1, model_path

    
    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, penal = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                if self.opt.losstype is not None:
                    loss = criterion(outputs, targets) + penal
                else:
                    loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()
                
                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            os.makedirs('./state_dict', exist_ok=True)
                            model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name, self.opt.dataset, test_acc, f1)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))
        return max_test_acc, max_f1, model_path
    
    def _evaluate(self, show_results=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                input_cols = self.student_input_cols if self.opt.model_name == 'ssegcnbertstudent' else self.opt.inputs_cols
                inputs = [sample_batched[col].to(self.opt.device) for col in input_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs, penal = self.model(inputs)
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1

        return test_acc, f1

    def _test(self):
        if self.best_model is None:
            self.best_model = copy.deepcopy(self.model)
        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)
        
    
    def run(self):
        criterion = nn.CrossEntropyLoss()
        if self.opt.model_name == 'ssegcnbertstudent':
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.student_lr, weight_decay=self.opt.l2reg)
        elif 'bert' not in self.opt.model_name:
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        else:
            optimizer = self.get_bert_optimizer(self.model)
        max_test_acc_overall = 0
        max_f1_overall = 0
        model_path = ''  # Initialize model_path
        if self.opt.model_name == 'ssegcnbertstudent':
            self._reset_params()
            max_test_acc, max_f1, model_path = self._train_kd(criterion, optimizer, max_test_acc_overall)
        elif 'bert' not in self.opt.model_name:
            self._reset_params()
            max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)
        else:
            max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)
        logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        if model_path and hasattr(self, 'best_model'):
            torch.save(self.best_model.state_dict(), model_path)
        logger.info('>> saved: {}'.format(model_path))
        logger.info('#' * 60)
        logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
        logger.info('max_f1_overall:{}'.format(max_f1_overall))
        self._test()


def main():
    model_classes = {
        'ssegcn': SSEGCNClassifier,
        'ssegcnbert': SSEGCNBertClassifier,
        'ssegcnbertstudent': SSEGCNStudentClassifier,

    }
    
    dataset_files = {
        'restaurant': {
            'train': './dataset/Restaurants_corenlp/train_write.json',
            'test': './dataset/Restaurants_corenlp/test_write.json',
        },
        'laptop': {
            'train': './dataset/Laptops_corenlp/train_write.json',
            'test': './dataset/Laptops_corenlp/test_write.json'
        },
        'twitter': {
            'train': './dataset/Tweets_corenlp/train_write.json',
            'test': './dataset/Tweets_corenlp/test_write.json',
        }
    }
    
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    
    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad, 
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax, 
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }
    
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ssegcn', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='laptop', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=50, help='GCN mem dim.')
    parser.add_argument('--num_layers', type=int, default=1, help='Num of GCN layers.')
    parser.add_argument('--polarities_dim', default=3, type=int, help='3')

    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--direct', default=False, help='directed graph or undirected graph')
    parser.add_argument('--loop', default=True)

    parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')
    
    parser.add_argument('--attention_heads', default=5, type=int, help='number of multi-attention heads')
    parser.add_argument('--max_length', default=85, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--vocab_dir', type=str, default='./dataset/Laptops_corenlp')
    parser.add_argument('--pad_id', default=0, type=int)
    parser.add_argument('--parseadj', default=False, action='store_true', help='dependency probability')
    parser.add_argument('--parsehead', default=False, action='store_true', help='dependency tree')
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--losstype', default=None, type=str, help="['doubleloss', 'orthogonalloss', 'differentiatedloss']")
    parser.add_argument('--alpha', default=0.25, type=float)
    parser.add_argument('--beta', default=0.25, type=float)
    
    # * bert
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
    parser.add_argument('--diff_lr', default=False, action='store_true')
    parser.add_argument('--bert_lr', default=2e-5, type=float)
    parser.add_argument('--teacher_path', default=None, type=str)
    parser.add_argument('--teacher_feature_dim', default=100, type=int)
    parser.add_argument('--kd_temperature', default=4.0, type=float)
    parser.add_argument('--kd_temperature_schedule', default='constant', type=str, choices=['constant', 'linear', 'cosine'])
    parser.add_argument('--kd_temperature_start', default=8.0, type=float)
    parser.add_argument('--kd_temperature_end', default=4.0, type=float)
    parser.add_argument('--kd_alpha', default=0.4, type=float)
    parser.add_argument('--kd_beta', default=0.4, type=float)
    parser.add_argument('--kd_gamma', default=0.2, type=float)
    parser.add_argument('--kd_warmup_epochs', default=0, type=int)
    parser.add_argument('--kd_ramp_epochs', default=1, type=int)
    parser.add_argument('--kd_scale_schedule', default='linear', type=str, choices=['linear', 'cosine'])
    parser.add_argument('--kd_logit_mode', default='kl', type=str, choices=['kl', 'dkd', 'dist'])
    parser.add_argument('--kd_logit_standardize', default=False, type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'y'))
    parser.add_argument('--kd_dkd_target_weight', default=1.0, type=float)
    parser.add_argument('--kd_dkd_non_target_weight', default=4.0, type=float)
    parser.add_argument('--kd_dist_inter_weight', default=1.0, type=float)
    parser.add_argument('--kd_dist_intra_weight', default=1.0, type=float)
    parser.add_argument('--kd_feature_mode', default='classifier_projection', type=str, choices=['classifier_projection', 'teacher_hidden'])
    parser.add_argument('--kd_feature_loss', default='mse', type=str, choices=['mse', 'cosine'])
    parser.add_argument('--kd_relation_weight', default=0.0, type=float)
    parser.add_argument('--kd_relation_angle_weight', default=2.0, type=float)
    parser.add_argument('--kd_contrastive_weight', default=0.0, type=float)
    parser.add_argument('--kd_contrastive_temperature', default=0.2, type=float)
    parser.add_argument('--kd_margin_weight', default=0.0, type=float)
    parser.add_argument('--kd_rank_weight', default=0.0, type=float)
    parser.add_argument('--kd_rank_temperature', default=1.0, type=float)
    parser.add_argument('--kd_proto_weight', default=0.0, type=float)
    parser.add_argument('--kd_proto_relation_weight', default=0.0, type=float)
    parser.add_argument('--kd_use_instance_weighting', default=True, type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'y'))
    parser.add_argument('--kd_min_weight', default=0.1, type=float)
    parser.add_argument('--kd_normalize_weights', default=True, type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'y'))
    parser.add_argument('--student_hidden_dim', default=32, type=int)
    parser.add_argument('--student_pos_dim', default=8, type=int)
    parser.add_argument('--student_post_dim', default=8, type=int)
    parser.add_argument('--student_input_dropout', default=0.2, type=float)
    parser.add_argument('--student_output_dropout', default=0.2, type=float)
    parser.add_argument('--student_lr', default=1e-3, type=float)
    parser.add_argument('--student_freeze_word_emb', default=True, type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'y'))
    opt = parser.parse_args()
    	
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = INPUT_COLSES[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    print("choice cuda:{}".format(opt.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)
    
    # set random seed
    setup_seed(opt.seed)

    if not os.path.exists('./log'):
        os.makedirs('./log', mode=0o777)
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./log', log_file)))

    ins = Instructor(opt)
    ins.run()

if __name__ == '__main__':
    main()
