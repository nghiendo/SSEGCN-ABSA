import os
import tempfile
import unittest

import numpy as np

from kg3_node2vec_utils import build_kg3_feature_tensor, load_node2vec_embeddings


class KG3Node2VecUtilsTests(unittest.TestCase):
    def test_build_kg3_feature_tensor_concats_word_and_aspect_vectors(self):
        embeddings = {
            'aspect::battery life': np.asarray([0.5, 0.6], dtype='float32'),
            'word::great': np.asarray([0.1, 0.2], dtype='float32'),
            'word::battery': np.asarray([0.3, 0.4], dtype='float32'),
        }

        features = build_kg3_feature_tensor(
            text_list=['great', 'battery'],
            aspect_node='aspect::battery life',
            tok2ori_map=[0, 1],
            term_tok2ori_map=[1],
            context_len=2,
            term_len=1,
            max_seq_len=8,
            node2vec_embeddings=embeddings,
            node2vec_dim=2,
        )

        np.testing.assert_allclose(features[1], [0.1, 0.2, 0.5, 0.6])
        np.testing.assert_allclose(features[2], [0.3, 0.4, 0.5, 0.6])
        np.testing.assert_allclose(features[4], [0.3, 0.4, 0.5, 0.6])

    def test_load_node2vec_embeddings_supports_index_keys(self):
        payload = '{"embeddings": {"0": [0.1, 0.2], "1": [0.3, 0.4]}}'
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as handle:
            handle.write(payload)
            temp_path = handle.name
        try:
            vectors = load_node2vec_embeddings(
                temp_path,
                index_to_node_id={0: 'aspect::service', 1: 'word::staff'},
                expected_dim=2,
            )
            np.testing.assert_allclose(vectors['aspect::service'], [0.1, 0.2])
            np.testing.assert_allclose(vectors['word::staff'], [0.3, 0.4])
        finally:
            os.remove(temp_path)

    def test_load_node2vec_embeddings_rejects_kg3_dataset_payload(self):
        payload = '{"dataset": "laptop", "records": []}'
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as handle:
            handle.write(payload)
            temp_path = handle.name
        try:
            with self.assertRaisesRegex(ValueError, 'Did you pass a KG3 dataset file'):
                load_node2vec_embeddings(temp_path, expected_dim=2)
        finally:
            os.remove(temp_path)


if __name__ == '__main__':
    unittest.main()
