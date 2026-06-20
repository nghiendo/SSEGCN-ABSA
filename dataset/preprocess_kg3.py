import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


DATASET_DIRS = {
    "laptop": "Laptops_corenlp",
    "restaurant": "Restaurants_corenlp",
    "twitter": "Tweets_corenlp",
}

POLARITIES = ("positive", "negative", "neutral")


def normalize_tokens(tokens: Iterable[str]) -> str:
    return " ".join(token.lower() for token in tokens).strip()


def aspect_term_from_record(record: Dict, aspect: Dict) -> str:
    term = normalize_tokens(aspect.get("term", []))
    if term:
        return term
    start, end = aspect["from"], aspect["to"]
    return normalize_tokens(record["token"][start:end])


def build_sentence_kg3(record: Dict) -> List[Dict]:
    tokens = record["token"]
    sentence_graphs = []

    for aspect in record.get("aspects", []):
        aspect_term = aspect_term_from_record(record, aspect)
        polarity = aspect["polarity"]
        aspect_span = set(range(aspect["from"], aspect["to"]))
        aspect_node = f"aspect::{aspect_term}"
        edges = []

        for token_index, token in enumerate(tokens):
            if token_index in aspect_span:
                continue
            token_text = token.lower()
            edges.append(
                {
                    "source": aspect_node,
                    "target": f"word::{token_text}",
                    "target_text": token_text,
                    "target_index": token_index,
                    "sentiment": polarity,
                }
            )

        sentence_graphs.append(
            {
                "aspect_term": aspect_term,
                "aspect_span": [aspect["from"], aspect["to"]],
                "polarity": polarity,
                "aspect_node": aspect_node,
                "edges": edges,
            }
        )

    return sentence_graphs


def update_split_graph(
    record: Dict,
    split: str,
    sentence_id: int,
    nodes: Dict[str, Dict],
    all_edges: Counter,
    sentiment_edges: Dict[str, Counter],
    allowed_nodes: Optional[Set[str]] = None,
) -> Dict:
    kg3 = build_sentence_kg3(record)
    augmented_record = dict(record)
    filtered_graphs = []
    kept_edges = 0
    dropped_edges = 0

    for aspect_graph in kg3:
        aspect_node = aspect_graph["aspect_node"]
        aspect_term = aspect_graph["aspect_term"]
        polarity = aspect_graph["polarity"]

        nodes.setdefault(
            aspect_node,
            {
                "node_id": aspect_node,
                "node_type": "aspect_term",
                "text": aspect_term,
            },
        )

        filtered_edges = []
        for edge in aspect_graph["edges"]:
            target = edge["target"]
            nodes.setdefault(
                target,
                {
                    "node_id": target,
                    "node_type": "word",
                    "text": edge["target_text"],
                },
            )

            if allowed_nodes is None or (aspect_node in allowed_nodes and target in allowed_nodes):
                filtered_edges.append(edge)
                edge_key = (aspect_node, target, polarity)
                all_edges[edge_key] += 1
                sentiment_edges[polarity][(aspect_node, target)] += 1
                kept_edges += 1
            else:
                dropped_edges += 1

        filtered_graph = dict(aspect_graph)
        filtered_graph["edges"] = filtered_edges
        filtered_graph["dropped_edges"] = len(aspect_graph["edges"]) - len(filtered_edges)
        filtered_graphs.append(filtered_graph)

    augmented_record["kg3"] = filtered_graphs
    augmented_record["kg3_meta"] = {
        "split": split,
        "sentence_id": sentence_id,
        "known_edges": kept_edges,
        "dropped_edges": dropped_edges,
        "uses_train_node_vocab_only": allowed_nodes is not None,
    }
    return augmented_record


def build_node_index(nodes: Dict[str, Dict]) -> Dict[str, int]:
    return {node_id: index for index, node_id in enumerate(sorted(nodes))}


def serialize_labelled_edges(edge_counter: Counter, node_to_index: Dict[str, int]) -> List[Dict]:
    return [
        {
            "source": source,
            "target": target,
            "source_index": node_to_index[source],
            "target_index": node_to_index[target],
            "sentiment": polarity,
            "weight": weight,
        }
        for (source, target, polarity), weight in sorted(edge_counter.items())
        if source in node_to_index and target in node_to_index
    ]


def summarize_nodes(node_to_index: Dict[str, int], nodes: Dict[str, Dict]) -> List[Dict]:
    return [
        {
            "index": node_to_index[node_id],
            **nodes[node_id],
        }
        for node_id in sorted(nodes)
        if node_id in node_to_index
    ]


def split_records(dataset_dir: Path, split: str) -> List[Dict]:
    input_path = dataset_dir / f"{split}.json"
    return json.loads(input_path.read_text(encoding="utf-8"))


def build_split_artifacts(
    dataset_name: str,
    split: str,
    records: List[Dict],
    allowed_nodes: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Dict], Counter, Dict[str, Counter], List[Dict], Dict]:
    nodes: Dict[str, Dict] = {}
    all_edges: Counter = Counter()
    sentiment_edges = {polarity: Counter() for polarity in POLARITIES}
    augmented_records = []
    total_aspects = 0

    for sentence_id, record in enumerate(records):
        augmented_records.append(
            update_split_graph(
                record=record,
                split=split,
                sentence_id=sentence_id,
                nodes=nodes,
                all_edges=all_edges,
                sentiment_edges=sentiment_edges,
                allowed_nodes=allowed_nodes,
            )
        )
        total_aspects += len(record.get("aspects", []))

    stats = {
        "dataset": dataset_name,
        "split": split,
        "sentences": len(records),
        "aspects": total_aspects,
        "nodes": len(nodes),
        "labelled_edges": len(all_edges),
        "positive_edges": len(sentiment_edges["positive"]),
        "negative_edges": len(sentiment_edges["negative"]),
        "neutral_edges": len(sentiment_edges["neutral"]),
        "uses_train_node_vocab_only": allowed_nodes is not None,
    }
    return nodes, all_edges, sentiment_edges, augmented_records, stats


def build_output_payload(
    dataset_name: str,
    split: str,
    nodes: Dict[str, Dict],
    all_edges: Counter,
    sentiment_edges: Dict[str, Counter],
    records: List[Dict],
    stats: Dict,
    node_to_index: Dict[str, int],
    train_vocab_size: Optional[int] = None,
) -> Dict:
    payload = {
        "dataset": dataset_name,
        "split": split,
        "stats": stats,
        "node_vocab": summarize_nodes(node_to_index, nodes),
        "graph": {
            "labelled_edges": serialize_labelled_edges(all_edges, node_to_index),
            "polarity_edge_counts": {
                polarity: len(sentiment_edges[polarity]) for polarity in POLARITIES
            },
        },
        "records": records,
    }
    if train_vocab_size is not None:
        payload["train_vocab_size"] = train_vocab_size
    return payload


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def process_dataset(dataset_name: str, dataset_dir: Path) -> None:
    train_records = split_records(dataset_dir, "train")
    test_records = split_records(dataset_dir, "test")

    train_nodes, train_edges, train_sentiment_edges, train_augmented, train_stats = build_split_artifacts(
        dataset_name=dataset_name,
        split="train",
        records=train_records,
    )
    train_node_to_index = build_node_index(train_nodes)
    train_payload = build_output_payload(
        dataset_name=dataset_name,
        split="train",
        nodes=train_nodes,
        all_edges=train_edges,
        sentiment_edges=train_sentiment_edges,
        records=train_augmented,
        stats=train_stats,
        node_to_index=train_node_to_index,
    )
    write_json(dataset_dir / "train_kg3.json", train_payload)

    test_nodes, test_edges, test_sentiment_edges, test_augmented, test_stats = build_split_artifacts(
        dataset_name=dataset_name,
        split="test",
        records=test_records,
        allowed_nodes=set(train_node_to_index),
    )
    test_stats["train_vocab_nodes"] = len(train_node_to_index)
    test_stats["observed_test_nodes"] = len(test_nodes)
    test_payload = build_output_payload(
        dataset_name=dataset_name,
        split="test",
        nodes=test_nodes,
        all_edges=test_edges,
        sentiment_edges=test_sentiment_edges,
        records=test_augmented,
        stats=test_stats,
        node_to_index=train_node_to_index,
        train_vocab_size=len(train_node_to_index),
    )
    write_json(dataset_dir / "test_kg3.json", test_payload)

    print(
        f"[KG3] {dataset_name}: train_edges={len(train_edges)}, test_edges={len(test_edges)}, "
        f"train_nodes={len(train_node_to_index)}, observed_test_nodes={len(test_nodes)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact leakage-safe KG3 artefacts for ABSA datasets.")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *DATASET_DIRS.keys()],
        help="Dataset to preprocess.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    datasets = DATASET_DIRS.items() if args.dataset == "all" else [(args.dataset, DATASET_DIRS[args.dataset])]
    for dataset_name, dataset_folder in datasets:
        process_dataset(dataset_name, base_dir / dataset_folder)


if __name__ == "__main__":
    main()
