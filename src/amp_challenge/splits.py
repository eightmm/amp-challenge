from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass

from Levenshtein import distance


@dataclass(frozen=True, slots=True)
class SequenceCluster:
    cluster_id: str
    representative: str
    members: tuple[str, ...]


def global_edit_identity(first: str, second: str, *, score_cutoff: int | None = None) -> float:
    denominator = max(len(first), len(second))
    if denominator == 0:
        return 1.0
    edit_distance = distance(first, second, score_cutoff=score_cutoff)
    if score_cutoff is not None and edit_distance > score_cutoff:
        return 0.0
    return 1.0 - edit_distance / denominator


def single_linkage_global_edit_clusters(
    sequences: list[str] | tuple[str, ...],
    *,
    identity_threshold: float,
    min_coverage: float,
) -> list[SequenceCluster]:
    """Build connected components so no qualifying pair can cross clusters."""
    if not 0.0 < identity_threshold <= 1.0:
        raise ValueError("identity_threshold must be in (0, 1]")
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    ordered = sorted(set(sequences))
    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parents[right_root] = left_root
        else:
            parents[left_root] = right_root

    for left_index, first in enumerate(ordered):
        for right_index in range(left_index + 1, len(ordered)):
            second = ordered[right_index]
            longer = max(len(first), len(second))
            if min(len(first), len(second)) / longer < min_coverage:
                continue
            max_distance = math.floor((1.0 - identity_threshold) * longer + 1e-12)
            edit_distance = distance(first, second, score_cutoff=max_distance)
            if edit_distance <= max_distance:
                union(left_index, right_index)

    components: dict[int, list[str]] = defaultdict(list)
    for index, sequence in enumerate(ordered):
        components[find(index)].append(sequence)

    clusters: list[SequenceCluster] = []
    for component in components.values():
        component_members = tuple(sorted(component))
        representative = min(
            component_members,
            key=lambda sequence: (-len(sequence), sequence),
        )
        digest = hashlib.sha256(representative.encode("ascii")).hexdigest()[:16]
        clusters.append(
            SequenceCluster(
                cluster_id=f"cluster-{digest}",
                representative=representative,
                members=component_members,
            )
        )
    return sorted(clusters, key=lambda cluster: cluster.cluster_id)


def audit_cross_fold_similarity(
    assignments: dict[str, int],
    *,
    identity_threshold: float,
    min_coverage: float,
) -> dict[str, float | int]:
    sequences = sorted(assignments)
    compared_pairs = 0
    violating_pairs = 0
    max_identity = 0.0
    for left_index, first in enumerate(sequences):
        for second in sequences[left_index + 1 :]:
            if assignments[first] == assignments[second]:
                continue
            longer = max(len(first), len(second))
            if min(len(first), len(second)) / longer < min_coverage:
                continue
            compared_pairs += 1
            identity = 1.0 - distance(first, second) / longer
            max_identity = max(max_identity, identity)
            if identity + 1e-12 >= identity_threshold:
                violating_pairs += 1
    return {
        "compared_cross_fold_pairs": compared_pairs,
        "violating_pairs": violating_pairs,
        "max_cross_fold_identity": round(max_identity, 12),
        "identity_threshold": identity_threshold,
        "min_coverage": min_coverage,
    }


def assign_cluster_folds(
    clusters: list[SequenceCluster],
    *,
    task_weights: dict[str, tuple[int, int]],
    folds: int,
    seed: int,
) -> dict[str, int]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    totals = [0, 0, 0]
    cluster_weights: dict[str, tuple[int, int, int]] = {}
    for cluster in clusters:
        mic = sum(task_weights.get(sequence, (0, 0))[0] for sequence in cluster.members)
        hc50 = sum(task_weights.get(sequence, (0, 0))[1] for sequence in cluster.members)
        weights = (mic, hc50, len(cluster.members))
        cluster_weights[cluster.cluster_id] = weights
        totals = [left + right for left, right in zip(totals, weights, strict=True)]

    targets = [max(total / folds, 1.0) for total in totals]
    loads = [[0, 0, 0] for _ in range(folds)]

    def priority(cluster: SequenceCluster) -> tuple[float, int, str]:
        weights = cluster_weights[cluster.cluster_id]
        normalized = sum(weight / target for weight, target in zip(weights, targets, strict=True))
        return (-normalized, -len(cluster.members), cluster.cluster_id)

    assignments: dict[str, int] = {}
    for cluster in sorted(clusters, key=priority):
        weights = cluster_weights[cluster.cluster_id]

        def fold_score(
            fold: int,
            current_weights: tuple[int, int, int] = weights,
            cluster_id: str = cluster.cluster_id,
        ) -> tuple[float, str]:
            prospective = [loads[fold][i] + current_weights[i] for i in range(3)]
            imbalance = sum(
                (value / target) ** 2 for value, target in zip(prospective, targets, strict=True)
            )
            tie = hashlib.sha256(f"{seed}:{cluster_id}:{fold}".encode("ascii")).hexdigest()
            return imbalance, tie

        selected = min(range(folds), key=fold_score)
        loads[selected] = [
            left + right for left, right in zip(loads[selected], weights, strict=True)
        ]
        for sequence in cluster.members:
            assignments[sequence] = selected
    return assignments


def cluster_lookup(clusters: list[SequenceCluster]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for cluster in clusters:
        for sequence in cluster.members:
            if sequence in lookup:
                raise ValueError(f"sequence occurs in multiple clusters: {sequence}")
            lookup[sequence] = cluster.cluster_id
    return lookup


def fold_task_counts(
    assignments: dict[str, int], task_weights: dict[str, tuple[int, int]]
) -> dict[int, dict[str, int]]:
    counts: dict[int, dict[str, int]] = defaultdict(
        lambda: {"sequences": 0, "mic_measurements": 0, "hc50_measurements": 0}
    )
    for sequence, fold in assignments.items():
        mic, hc50 = task_weights.get(sequence, (0, 0))
        counts[fold]["sequences"] += 1
        counts[fold]["mic_measurements"] += mic
        counts[fold]["hc50_measurements"] += hc50
    return dict(sorted(counts.items()))
