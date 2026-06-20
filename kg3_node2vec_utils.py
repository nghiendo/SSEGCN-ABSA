import json
import os
from pathlib import Path

import numpy as np


def _load_json_root(data_path):
    with open(data_path, encoding='utf-8') as infile:
        return json.load(infile)


def parse_kg3_data(data_path):
    root = _load_json_root(data_path)
    records = root['records'] if isinstance(root, dict) else root
    all_data = []
    for sentence in records:
        aspects = sentence.get('aspects', [])
        kg3_graphs = sentence.get('kg3', [])
        if len(aspects) != len(kg3_graphs):
            raise ValueError('KG3 aspect count mismatch in {}'.format(data_path))
        for aspect, kg3_graph in zip(aspects, kg3_graphs):
            all_data.append({
                'aspect': aspect,
                'kg3': kg3_graph,
                'text_list': list(sentence['token']),
            })
    return all_data


def load_kg3_node_vocab(kg3_path):
    root = _load_json_root(kg3_path)
    node_vocab = root.get('node_vocab', [])
    index_to_node_id = {}
    node_id_to_index = {}
    for node in node_vocab:
        index_to_node_id[node['index']] = node['node_id']
        node_id_to_index[node['node_id']] = node['index']
    return index_to_node_id, node_id_to_index


def load_node2vec_embeddings(embedding_path, index_to_node_id=None, expected_dim=None):
    if embedding_path is None:
        return {}
    if not os.path.exists(embedding_path):
        raise FileNotFoundError('Node2Vec embedding file not found: {}'.format(embedding_path))

    suffix = Path(embedding_path).suffix.lower()
    vectors = {}

    def register_vector(raw_key, values):
        if expected_dim is not None and len(values) != expected_dim:
            raise ValueError('Node2Vec dimension mismatch for {}: expected {}, got {}'.format(raw_key, expected_dim, len(values)))
        node_id = None
        if index_to_node_id is not None:
            try:
                raw_index = int(raw_key)
                node_id = index_to_node_id.get(raw_index)
            except (TypeError, ValueError):
                node_id = None
        if node_id is None:
            node_id = str(raw_key)
        vectors[node_id] = np.asarray(values, dtype='float32')

    if suffix == '.json':
        raw = _load_json_root(embedding_path)
        if isinstance(raw, dict) and 'embeddings' in raw:
            raw = raw['embeddings']
        if isinstance(raw, dict):
            for key, values in raw.items():
                register_vector(key, values)
        elif isinstance(raw, list):
            for item in raw:
                key = item.get('node_id', item.get('index'))
                values = item.get('vector')
                register_vector(key, values)
        else:
            raise ValueError('Unsupported Node2Vec JSON format: {}'.format(embedding_path))
    else:
        with open(embedding_path, 'r', encoding='utf-8') as infile:
            first_line = True
            for line in infile:
                tokens = line.strip().split()
                if not tokens:
                    continue
                if first_line and len(tokens) == 2:
                    try:
                        int(tokens[0])
                        int(tokens[1])
                        first_line = False
                        continue
                    except ValueError:
                        pass
                first_line = False
                register_vector(tokens[0], [float(v) for v in tokens[1:]])
    return vectors


def build_kg3_feature_tensor(text_list, aspect_node, tok2ori_map, term_tok2ori_map, context_len, term_len, max_seq_len, node2vec_embeddings, node2vec_dim):
    feature_dim = node2vec_dim * 2
    features = np.zeros((max_seq_len, feature_dim), dtype='float32')
    if node2vec_dim <= 0:
        return features

    zero_vec = np.zeros(node2vec_dim, dtype='float32')
    aspect_vec = node2vec_embeddings.get(aspect_node, zero_vec)

    def token_feature(ori_index):
        word_node = 'word::{}'.format(text_list[ori_index].lower())
        word_vec = node2vec_embeddings.get(word_node, zero_vec)
        return np.concatenate([word_vec, aspect_vec], axis=0)

    for bert_index, ori_index in enumerate(tok2ori_map, start=1):
        if bert_index >= max_seq_len:
            break
        features[bert_index] = token_feature(ori_index)

    term_offset = context_len + 2
    for i, ori_index in enumerate(term_tok2ori_map):
        bert_index = term_offset + i
        if bert_index >= max_seq_len:
            break
        features[bert_index] = token_feature(ori_index)

    return features
