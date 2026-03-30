"""Build reasoning trees from sampled math solutions by merging semantically similar steps."""

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

INPUT_PATH = "results/math_filtered.jsonl"
OUTPUT_PATH = "results/reasoning_trees.jsonl"
MODEL_NAME = "bert-base-uncased"
SIMILARITY_THRESHOLD = 0.85
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class TreeNode:
    node_id: int
    depth: int
    text: str
    generation_ids: list[int] = field(default_factory=list)
    children: list["TreeNode"] = field(default_factory=list)
    parent_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "text": self.text,
            "generation_ids": self.generation_ids,
            "children": [c.to_dict() for c in self.children],
        }


# ── Embedding ────────────────────────────────────────────────────────────────


class StepEmbedder:
    """Mean-pooled sentence embeddings via a HuggingFace encoder model."""

    def __init__(self, model_name: str, device: str, batch_size: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        all_embs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            output = self.model(**encoded)
            # mean pool over non-padding tokens
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            emb = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            # L2 normalize
            emb = emb / emb.norm(dim=1, keepdim=True)
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)


# ── Clustering ───────────────────────────────────────────────────────────────


def cluster_steps(
    texts: list[str],
    generation_ids: list[int],
    embeddings: np.ndarray,
    threshold: float,
) -> list[tuple[str, list[int], np.ndarray]]:
    """Greedy clustering. Returns list of (representative_text, [gen_ids], centroid)."""
    clusters: list[tuple[str, list[int], np.ndarray]] = []

    for i, (text, gen_id, emb) in enumerate(zip(texts, generation_ids, embeddings)):
        best_sim = -1.0
        best_idx = -1
        for j, (_, _, centroid) in enumerate(clusters):
            sim = float(np.dot(emb, centroid))
            if sim > best_sim:
                best_sim = sim
                best_idx = j

        if best_sim >= threshold and best_idx >= 0:
            rep_text, gids, centroid = clusters[best_idx]
            gids.append(gen_id)
            # update centroid as running mean, re-normalize
            new_centroid = centroid * (len(gids) - 1) / len(gids) + emb / len(gids)
            new_centroid = new_centroid / np.linalg.norm(new_centroid)
            clusters[best_idx] = (rep_text, gids, new_centroid)
        else:
            clusters.append((text, [gen_id], emb.copy()))

    return clusters


# ── Tree building ────────────────────────────────────────────────────────────


def split_steps(generation: str) -> list[str]:
    return [s.strip() for s in generation.split("\n\n") if s.strip()]


def build_tree(
    problem: str,
    generations: list[str],
    embedder: StepEmbedder,
    threshold: float,
) -> tuple[TreeNode, dict[int, TreeNode], list[int]]:
    """Build reasoning tree for one problem.

    Returns (root, node_lookup, generation_leaf_ids).
    """
    node_counter = 0
    node_lookup: dict[int, TreeNode] = {}

    # Root node
    root = TreeNode(node_id=node_counter, depth=0, text=problem,
                    generation_ids=list(range(len(generations))))
    node_lookup[node_counter] = root
    node_counter += 1

    # Split all generations into steps
    all_steps = [split_steps(gen) for gen in generations]
    max_depth = max(len(s) for s in all_steps)

    # Track which node each generation is currently at
    gen_current_node = {i: root.node_id for i in range(len(generations))}
    # Track leaf node for each generation (updated as we go deeper)
    gen_leaf = {i: root.node_id for i in range(len(generations))}

    for d in range(max_depth):
        # Collect (parent_node_id, gen_id, step_text) for gens that have a step at depth d
        pending: dict[int, list[tuple[int, str]]] = {}  # parent_id -> [(gen_id, text)]
        for gen_id, steps in enumerate(all_steps):
            if d < len(steps):
                parent_id = gen_current_node[gen_id]
                pending.setdefault(parent_id, []).append((gen_id, steps[d]))

        if not pending:
            break

        # Collect all unique texts for batch embedding
        all_texts = []
        text_indices: dict[int, list[tuple[int, int]]] = {}  # parent_id -> [(local_idx, global_idx)]
        for parent_id, items in pending.items():
            text_indices[parent_id] = []
            for gen_id, text in items:
                text_indices[parent_id].append((gen_id, len(all_texts)))
                all_texts.append(text)

        if not all_texts:
            break

        embeddings = embedder.embed(all_texts)

        # Cluster per parent
        for parent_id, items in pending.items():
            indices = text_indices[parent_id]
            group_texts = [all_texts[gi] for _, gi in indices]
            group_gen_ids = [gen_id for gen_id, _ in indices]
            group_embs = np.stack([embeddings[gi] for _, gi in indices])

            clusters = cluster_steps(group_texts, group_gen_ids, group_embs, threshold)
            parent_node = node_lookup[parent_id]

            for rep_text, gids, _ in clusters:
                child = TreeNode(
                    node_id=node_counter,
                    depth=d + 1,
                    text=rep_text,
                    generation_ids=sorted(gids),
                    parent_id=parent_id,
                )
                node_lookup[node_counter] = child
                parent_node.children.append(child)
                node_counter += 1

                for gid in gids:
                    gen_current_node[gid] = child.node_id
                    gen_leaf[gid] = child.node_id

    generation_leaf_ids = [gen_leaf[i] for i in range(len(generations))]
    return root, node_lookup, generation_leaf_ids


# ── Tree distance ────────────────────────────────────────────────────────────


def get_ancestors(node_lookup: dict[int, TreeNode], node_id: int) -> list[int]:
    """Return path from node to root (inclusive)."""
    path = []
    while node_id is not None:
        path.append(node_id)
        node_id = node_lookup[node_id].parent_id
    return path


def tree_distance(node_lookup: dict[int, TreeNode], id_a: int, id_b: int) -> int:
    """Compute tree path distance between two nodes via LCA."""
    ancestors_a = get_ancestors(node_lookup, id_a)
    ancestors_b = set(get_ancestors(node_lookup, id_b))

    for i, ancestor in enumerate(ancestors_a):
        if ancestor in ancestors_b:
            lca_id = ancestor
            depth_lca = node_lookup[lca_id].depth
            depth_a = node_lookup[id_a].depth
            depth_b = node_lookup[id_b].depth
            return (depth_a - depth_lca) + (depth_b - depth_lca)

    # Should not happen in a valid tree
    return -1


def compute_pairwise_distances(
    node_lookup: dict[int, TreeNode],
    generation_leaf_ids: list[int],
) -> list[list[int]]:
    n = len(generation_leaf_ids)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = tree_distance(node_lookup, generation_leaf_ids[i], generation_leaf_ids[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Build reasoning trees from sampled solutions.")
    parser.add_argument("--input", default=INPUT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    with open(args.input) as f:
        records = [json.loads(line) for line in f]
    print(f"Loaded {len(records)} problems from {args.input}")

    embedder = StepEmbedder(args.model, args.device, args.batch_size)
    print(f"Loaded embedding model: {args.model} on {args.device}")

    results = []
    total_nodes = 0
    total_max_depth = 0

    for i, record in enumerate(records):
        root, node_lookup, leaf_ids = build_tree(
            record["problem"], record["generations"], embedder, args.threshold
        )
        distances = compute_pairwise_distances(node_lookup, leaf_ids)

        result = {
            "problem": record["problem"],
            "pass_rate": record["pass_rate"],
            "tree": root.to_dict(),
            "num_nodes": len(node_lookup),
            "max_depth": max(n.depth for n in node_lookup.values()),
            "generation_leaf_ids": leaf_ids,
            "pairwise_distances": distances,
        }
        results.append(result)

        total_nodes += result["num_nodes"]
        total_max_depth = max(total_max_depth, result["max_depth"])

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(records)} problems")

    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    avg_nodes = total_nodes / len(results) if results else 0
    print(f"\nSaved {len(results)} reasoning trees to {args.output}")
    print(f"Avg nodes per tree: {avg_nodes:.1f}")
    print(f"Max tree depth: {total_max_depth}")


if __name__ == "__main__":
    main()
