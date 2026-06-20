import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


DATASET_DIRS = {
    "laptop": "Laptops_corenlp",
    "restaurant": "Restaurants_corenlp",
    "twitter": "Tweets_corenlp",
}


def load_kg3_graph(path: Path):
    with path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)
    node_vocab = payload["node_vocab"]
    labelled_edges = payload["graph"]["labelled_edges"]
    return node_vocab, labelled_edges


def build_adjacency(num_nodes, labelled_edges):
    adjacency = defaultdict(list)
    for edge in labelled_edges:
        src = int(edge["source_index"])
        dst = int(edge["target_index"])
        weight = max(float(edge.get("weight", 1.0)), 1.0)
        adjacency[src].append((dst, weight))
        adjacency[dst].append((src, weight))

    for node_index in range(num_nodes):
        adjacency[node_index] = adjacency.get(node_index, [])
    return adjacency


def weighted_sample(neighbors, rng):
    if not neighbors:
        return None
    if len(neighbors) == 1:
        return neighbors[0][0]

    total = sum(weight for _, weight in neighbors)
    threshold = rng.random() * total
    cumulative = 0.0
    for node_index, weight in neighbors:
        cumulative += weight
        if cumulative >= threshold:
            return node_index
    return neighbors[-1][0]


def generate_walks(adjacency, num_walks, walk_length, seed):
    rng = random.Random(seed)
    nodes = list(adjacency.keys())
    walks = []

    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start_node in nodes:
            if not adjacency[start_node]:
                continue
            walk = [start_node]
            current = start_node
            for _ in range(walk_length - 1):
                next_node = weighted_sample(adjacency[current], rng)
                if next_node is None:
                    break
                walk.append(next_node)
                current = next_node
            if len(walk) > 1:
                walks.append(walk)
    return walks


def build_skipgram_pairs(walks, window_size, max_pairs=None):
    pairs = []
    for walk in walks:
        for center_index, center_node in enumerate(walk):
            left = max(0, center_index - window_size)
            right = min(len(walk), center_index + window_size + 1)
            for context_index in range(left, right):
                if context_index == center_index:
                    continue
                pairs.append((center_node, walk[context_index]))
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs
    return pairs


def train_skipgram(
    num_nodes,
    pairs,
    embedding_dim,
    negative_samples,
    batch_size,
    epochs,
    learning_rate,
    adjacency,
    seed,
    device,
):
    if not pairs:
        raise ValueError("No skip-gram pairs were generated from the KG3 graph.")

    rng = np.random.default_rng(seed)
    degrees = np.asarray(
        [max(sum(weight for _, weight in adjacency[node]), 1.0) for node in range(num_nodes)],
        dtype=np.float64,
    )
    negative_distribution = np.power(degrees, 0.75)
    negative_distribution /= negative_distribution.sum()

    input_embeddings = torch.nn.Embedding(num_nodes, embedding_dim, device=device)
    output_embeddings = torch.nn.Embedding(num_nodes, embedding_dim, device=device)
    bound = 1.0 / max(embedding_dim, 1)
    torch.nn.init.uniform_(input_embeddings.weight, -bound, bound)
    torch.nn.init.zeros_(output_embeddings.weight)

    optimizer = torch.optim.Adam(
        list(input_embeddings.parameters()) + list(output_embeddings.parameters()),
        lr=learning_rate,
    )

    pair_array = np.asarray(pairs, dtype=np.int64)
    for epoch in range(epochs):
        rng.shuffle(pair_array)
        epoch_loss = 0.0

        for start in range(0, len(pair_array), batch_size):
            batch = pair_array[start:start + batch_size]
            centers = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            contexts = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
            negatives = rng.choice(
                num_nodes,
                size=(len(batch), negative_samples),
                replace=True,
                p=negative_distribution,
            )
            negatives = torch.tensor(negatives, dtype=torch.long, device=device)

            center_vecs = input_embeddings(centers)
            context_vecs = output_embeddings(contexts)
            negative_vecs = output_embeddings(negatives)

            positive_score = torch.sum(center_vecs * context_vecs, dim=1)
            positive_loss = F.logsigmoid(positive_score)

            negative_score = torch.bmm(negative_vecs, center_vecs.unsqueeze(2)).squeeze(2)
            negative_loss = F.logsigmoid(-negative_score).sum(dim=1)

            loss = -(positive_loss + negative_loss).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch)

        avg_loss = epoch_loss / len(pair_array)
        print(f"[Node2Vec] epoch={epoch + 1}/{epochs} loss={avg_loss:.4f}")

    embeddings = input_embeddings.weight.detach().cpu().numpy()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return embeddings / norms


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Node2Vec-style embeddings from train_kg3.json.")
    parser.add_argument("--dataset", choices=DATASET_DIRS.keys(), required=True, help="Dataset name.")
    parser.add_argument("--input", type=str, default=None, help="Path to train_kg3.json. Defaults by dataset.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path. Defaults by dataset.")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--num_walks", type=int, default=10, help="Random walks per node.")
    parser.add_argument("--walk_length", type=int, default=20, help="Length of each random walk.")
    parser.add_argument("--window_size", type=int, default=5, help="Skip-gram context window.")
    parser.add_argument("--negative_samples", type=int, default=5, help="Negative samples per positive pair.")
    parser.add_argument("--epochs", type=int, default=3, help="Skip-gram training epochs.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Optimizer learning rate.")
    parser.add_argument("--max_pairs", type=int, default=500000, help="Optional cap for generated skip-gram pairs.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda. Defaults to cuda when available.")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / DATASET_DIRS[args.dataset]
    input_path = Path(args.input) if args.input else dataset_dir / "train_kg3.json"
    output_path = Path(args.output) if args.output else dataset_dir / "node2vec_embeddings.json"

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    node_vocab, labelled_edges = load_kg3_graph(input_path)
    num_nodes = len(node_vocab)
    adjacency = build_adjacency(num_nodes, labelled_edges)
    walks = generate_walks(adjacency, args.num_walks, args.walk_length, args.seed)
    pairs = build_skipgram_pairs(walks, args.window_size, max_pairs=args.max_pairs)

    print(
        f"[Node2Vec] dataset={args.dataset} nodes={num_nodes} edges={len(labelled_edges)} "
        f"walks={len(walks)} pairs={len(pairs)} device={device}"
    )

    embeddings = train_skipgram(
        num_nodes=num_nodes,
        pairs=pairs,
        embedding_dim=args.dim,
        negative_samples=args.negative_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        adjacency=adjacency,
        seed=args.seed,
        device=device,
    )

    payload = {
        "dataset": args.dataset,
        "embedding_dim": args.dim,
        "input_path": str(input_path),
        "embeddings": {str(index): embeddings[index].round(8).tolist() for index in range(num_nodes)},
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[Node2Vec] wrote embeddings to {output_path}")


if __name__ == "__main__":
    main()
