from __future__ import annotations

import unittest

from amp_challenge.splits import (
    assign_cluster_folds,
    audit_cross_fold_similarity,
    cluster_lookup,
    single_linkage_global_edit_clusters,
)


class SplitTests(unittest.TestCase):
    def test_single_linkage_and_fold_assignment_are_deterministic(self) -> None:
        sequences = [
            "KWKLFKKIGAVLKVL",
            "KWKLFKKIGAVLKVA",
            "ACDEFGHIKLMNPQR",
            "WWRRWWRRWWRRWWR",
        ]
        first = single_linkage_global_edit_clusters(
            sequences,
            identity_threshold=0.8,
            min_coverage=0.8,
        )
        second = single_linkage_global_edit_clusters(
            list(reversed(sequences)),
            identity_threshold=0.8,
            min_coverage=0.8,
        )
        self.assertEqual(first, second)
        lookup = cluster_lookup(first)
        self.assertEqual(lookup[sequences[0]], lookup[sequences[1]])
        weights = {sequence: (index + 1, index % 2) for index, sequence in enumerate(sequences)}
        assignments = assign_cluster_folds(first, task_weights=weights, folds=3, seed=42)
        again = assign_cluster_folds(first, task_weights=weights, folds=3, seed=42)
        self.assertEqual(assignments, again)
        for cluster in first:
            self.assertEqual(len({assignments[member] for member in cluster.members}), 1)
        audit = audit_cross_fold_similarity(
            assignments,
            identity_threshold=0.8,
            min_coverage=0.8,
        )
        self.assertEqual(audit["violating_pairs"], 0)

    def test_single_linkage_closes_similarity_chains(self) -> None:
        sequences = ["AAAAAAAAAA", "AAAAAAAAAC", "AAAAAAAACC"]
        clusters = single_linkage_global_edit_clusters(
            sequences,
            identity_threshold=0.9,
            min_coverage=1.0,
        )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(set(clusters[0].members), set(sequences))


if __name__ == "__main__":
    unittest.main()
