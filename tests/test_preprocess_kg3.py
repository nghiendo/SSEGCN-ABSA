import unittest
from collections import Counter

from dataset.preprocess_kg3 import build_sentence_kg3, update_split_graph


class KG3PreprocessTests(unittest.TestCase):
    def test_build_sentence_graph_skips_aspect_tokens_and_keeps_polarity(self):
        record = {
            "token": ["good", "battery", "life", "overall"],
            "aspects": [
                {
                    "term": ["battery", "life"],
                    "from": 1,
                    "to": 3,
                    "polarity": "positive",
                }
            ],
        }

        graphs = build_sentence_kg3(record)

        self.assertEqual(len(graphs), 1)
        self.assertEqual(graphs[0]["aspect_node"], "aspect::battery life")
        self.assertEqual(
            graphs[0]["edges"],
            [
                {
                    "source": "aspect::battery life",
                    "target": "word::good",
                    "target_text": "good",
                    "target_index": 0,
                    "sentiment": "positive",
                },
                {
                    "source": "aspect::battery life",
                    "target": "word::overall",
                    "target_text": "overall",
                    "target_index": 3,
                    "sentiment": "positive",
                },
            ],
        )

    def test_update_split_graph_accumulates_labelled_edges(self):
        record = {
            "token": ["service", "and", "staff", "shine"],
            "aspects": [
                {
                    "term": ["service"],
                    "from": 0,
                    "to": 1,
                    "polarity": "positive",
                },
                {
                    "term": ["staff"],
                    "from": 2,
                    "to": 3,
                    "polarity": "positive",
                },
            ],
        }

        nodes = {}
        all_edges = Counter()
        sentiment_edges = {"positive": Counter(), "negative": Counter(), "neutral": Counter()}

        augmented = update_split_graph(
            record=record,
            split="train",
            sentence_id=0,
            nodes=nodes,
            all_edges=all_edges,
            sentiment_edges=sentiment_edges,
        )

        self.assertIn("kg3", augmented)
        self.assertEqual(len(augmented["kg3"]), 2)
        self.assertEqual(all_edges[("aspect::service", "word::staff", "positive")], 1)
        self.assertEqual(all_edges[("aspect::staff", "word::service", "positive")], 1)
        self.assertEqual(sentiment_edges["positive"][("aspect::service", "word::staff")], 1)
        self.assertIn("aspect::service", nodes)
        self.assertIn("word::shine", nodes)

    def test_test_split_drops_unseen_nodes_when_using_train_vocab(self):
        record = {
            "token": ["service", "felt", "premium"],
            "aspects": [
                {
                    "term": ["service"],
                    "from": 0,
                    "to": 1,
                    "polarity": "positive",
                }
            ],
        }

        nodes = {}
        all_edges = Counter()
        sentiment_edges = {"positive": Counter(), "negative": Counter(), "neutral": Counter()}
        allowed_nodes = {"aspect::service", "word::felt"}

        augmented = update_split_graph(
            record=record,
            split="test",
            sentence_id=0,
            nodes=nodes,
            all_edges=all_edges,
            sentiment_edges=sentiment_edges,
            allowed_nodes=allowed_nodes,
        )

        self.assertEqual(len(augmented["kg3"][0]["edges"]), 1)
        self.assertEqual(augmented["kg3"][0]["dropped_edges"], 1)
        self.assertEqual(all_edges[("aspect::service", "word::felt", "positive")], 1)
        self.assertNotIn(("aspect::service", "word::premium", "positive"), all_edges)


if __name__ == "__main__":
    unittest.main()
