"""
RigidGS point-label refinement. Relabels points, never removes them.

Two methods (preprocessing.py --point-refinement):

  knn (default, used for the paper):
      KNN voting -- each point takes the label of its K=40 nearest
      neighbours when at least 90% of them agree on a different label.

  dbscan:
      Step 1: DBSCAN per object  — relabels scattered/floating points as
              background
      Step 2: radius-based voting — fixes boundary bleeding between objects
      Distances are multiples of the mean inter-point spacing. More
      aggressive: on fern it moves ~190 thin-frond points to background.
"""

import numpy as np
from pathlib import Path
import json
import argparse
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, BallTree
import open3d as o3d
from tqdm import tqdm


class PointCloudRefiner:
    """
    Refine labeled point clouds by spatial coherence (method 'knn' or
    'dbscan', see the module docstring). Relabels misclassified points —
    DOES NOT remove any points.
    """

    def __init__(self, config: dict = None):
        """
        Args:
            config: Optional parameter overrides:
                - method:               'knn' (default) or 'dbscan'
                knn:
                - knn_neighbors:        K nearest neighbours that vote          (default: 40)
                - knn_threshold:        neighbour agreement fraction to relabel (default: 0.9)
                dbscan:
                - spacing_k:            K used to compute mean inter-point spacing (default: 6)
                - mean_spacing:         fixed spacing (scene units) instead of estimating it
                - dbscan_scale:         eps = mean_spacing * scale             (default: 3.0)
                - min_cluster_fraction: min cluster = max(10, N * fraction)   (default: 0.005)
                - vote_radius_scale:    radius = mean_spacing * scale         (default: 5.0)
                - vote_threshold:       neighbor agreement fraction to relabel (default: 0.7)
        """
        self.config = config or {}

        self.method                = self.config.get('method', 'knn')
        if self.method not in ('knn', 'dbscan'):
            raise ValueError(f"unknown point-refinement method {self.method!r}")
        self.knn_neighbors         = self.config.get('knn_neighbors', 40)
        self.knn_threshold         = self.config.get('knn_threshold', 0.9)

        self.spacing_k             = self.config.get('spacing_k', 6)
        self.mean_spacing          = self.config.get('mean_spacing', None)
        self.dbscan_scale          = self.config.get('dbscan_scale', 3.0)
        self.min_cluster_fraction  = self.config.get('min_cluster_fraction', 0.005)
        self.vote_radius_scale     = self.config.get('vote_radius_scale', 5.0)
        self.vote_threshold        = self.config.get('vote_threshold', 0.7)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine_labels(self, ply_path: str, metadata_path: str,
                      output_path: str, colmap_dir: str = None) -> dict:
        """
        Run the full refinement pipeline.

        Returns:
            Dict with stats (total_points, total_changed, per-step counts,
            auto-computed scene scale values).
        """
        print("\n" + "=" * 70)
        print(" " * 18 + f"POINT CLOUD REFINEMENT ({self.method})")
        print("=" * 70)
        print("All points are kept — only labels are changed.")
        print("=" * 70)

        points, colors, labels = self._load_labeled_points(ply_path, metadata_path)
        total_points = len(points)
        print(f"\nTotal points: {total_points:,}")

        original_labels = [l['object_name'] for l in labels]
        # object_id must follow object_name when a point is relabeled.
        name_to_id = {}
        for l in labels:
            name_to_id.setdefault(l['object_name'], l['object_id'])

        stats = {'total_points': total_points, 'method': self.method,
                 'relabeled_by_step': {}}
        steps = []

        if self.method == 'knn':
            print(f"\nKNN voting: K={self.knn_neighbors}, "
                  f"threshold {self.knn_threshold:.0%}...")
            labels, n_knn = self._knn_vote(points, labels)
            stats['relabeled_by_step']['knn_voting'] = n_knn
            steps.append(("KNN voting", n_knn))
            print(f"  Relabeled: {n_knn:,} points by neighborhood consensus")
        else:
            # ── Auto-compute scene scale ────────────────────────────────
            if self.mean_spacing is not None:
                print("\nUsing the given mean point spacing...")
                mean_spacing = float(self.mean_spacing)
            else:
                print("\nComputing scene scale from point density...")
                mean_spacing = self._compute_mean_spacing(points)

            dbscan_eps  = mean_spacing * self.dbscan_scale
            vote_radius = mean_spacing * self.vote_radius_scale

            print(f"  Mean point spacing : {mean_spacing * 100:.3f} cm")
            print(f"  DBSCAN eps         : {dbscan_eps * 100:.3f} cm  (×{self.dbscan_scale})")
            print(f"  Vote radius        : {vote_radius * 100:.3f} cm  (×{self.vote_radius_scale})")
            print(f"  Vote threshold     : {self.vote_threshold:.0%}")

            unique_objects = {l['object_name'] for l in labels} - {'background'}
            print(f"  Objects to clean   : {len(unique_objects)}")
            stats.update({'mean_spacing_m': float(mean_spacing),
                          'dbscan_eps_m':   float(dbscan_eps),
                          'vote_radius_m':  float(vote_radius)})

            # ── Step 1: DBSCAN per object ───────────────────────────────
            print("\n[Step 1/2] DBSCAN cluster cleaning (scattered / floating points)...")
            labels, n_dbscan = self._dbscan_per_object(
                points, labels, unique_objects, dbscan_eps
            )
            stats['relabeled_by_step']['dbscan_cleanup'] = n_dbscan
            steps.append(("DBSCAN cleanup", n_dbscan))
            print(f"  Relabeled: {n_dbscan:,} scattered points → background")

            # ── Step 2: Radius-based voting ─────────────────────────────
            print("\n[Step 2/2] Radius-based voting (boundary bleeding)...")
            labels, n_voted = self._radius_vote(points, labels, vote_radius)
            stats['relabeled_by_step']['radius_voting'] = n_voted
            steps.append(("Radius voting", n_voted))
            print(f"  Relabeled: {n_voted:,} boundary points by neighborhood consensus")

        for l in labels:
            l['object_id'] = name_to_id.get(l['object_name'], l['object_id'])

        # ── Summary ─────────────────────────────────────────────────────
        final_labels  = [l['object_name'] for l in labels]
        total_changed = sum(1 for o, f in zip(original_labels, final_labels) if o != f)
        stats['total_changed'] = total_changed

        print("\n" + "=" * 70)
        print("REFINEMENT SUMMARY")
        print("=" * 70)
        print(f"Total points:         {total_points:,} (ALL KEPT)")
        print(f"Labels changed:       {total_changed:,} ({100 * total_changed / total_points:.1f}%)")
        for step, n in steps:
            print(f"  → {step + ':':17s} {n:,}")
        print("=" * 70)

        self._print_per_object_stats(labels, original_labels)

        print(f"\nSaving to: {output_path}")
        self._save_refined_points(points, colors, labels, Path(output_path), colmap_dir)

        return stats

    # ------------------------------------------------------------------
    # Scene scale
    # ------------------------------------------------------------------

    def _compute_mean_spacing(self, points: np.ndarray) -> float:
        """
        Estimate the natural inter-point spacing by averaging the distance
        to the nearest neighbors over a random sample of the cloud.
        """
        sample_size = min(5000, len(points))
        idx    = np.random.default_rng(0).choice(len(points), sample_size, replace=False)
        sample = points[idx]

        nbrs = NearestNeighbors(n_neighbors=self.spacing_k, algorithm='kd_tree').fit(points)
        distances, _ = nbrs.kneighbors(sample)
        # column 0 is the point itself (distance = 0) when querying from the full set,
        # but since we query a sample against the full set column 0 is the nearest real neighbour.
        return float(distances[:, 1:].mean())

    # ------------------------------------------------------------------
    # knn — KNN voting
    # ------------------------------------------------------------------

    def _knn_vote(self, points, labels):
        """
        Visit the points in order; relabel a point when at least
        knn_threshold of its knn_neighbors nearest neighbours share a
        different label. Neighbours' labels are read as they stand, so a
        point relabeled earlier already votes with its new label.
        """
        nbrs = NearestNeighbors(n_neighbors=self.knn_neighbors + 1,
                                algorithm='kd_tree').fit(points)
        _, indices = nbrs.kneighbors(points)

        relabeled_count = 0
        for i in tqdm(range(len(points)), desc="KNN voting"):
            votes = {}
            for idx in indices[i][1:]:                  # column 0 is the point itself
                name = labels[idx]['object_name']
                votes[name] = votes.get(name, 0) + 1
            majority_label = max(votes, key=votes.get)
            majority_ratio = votes[majority_label] / self.knn_neighbors

            if (majority_ratio >= self.knn_threshold
                    and majority_label != labels[i]['object_name']):
                labels[i]['object_name'] = majority_label
                labels[i]['confidence']  = majority_ratio
                labels[i]['refined']     = 'knn_voting'
                relabeled_count += 1

        return labels, relabeled_count

    # ------------------------------------------------------------------
    # dbscan, step 1 — DBSCAN per object
    # ------------------------------------------------------------------

    def _dbscan_per_object(self, points, labels, objects, eps):
        """
        For each object: cluster its points with DBSCAN, then relabel any
        point that belongs to a small/isolated cluster as background.

        Keeps the largest cluster(s) that exceed min_cluster_size.
        If no cluster passes, the single biggest cluster is kept.
        """
        relabeled_count = 0

        for obj_name in tqdm(objects, desc="DBSCAN per object"):
            obj_mask    = np.array([l['object_name'] == obj_name for l in labels])
            obj_indices = np.where(obj_mask)[0]
            obj_points  = points[obj_mask]
            n_obj       = len(obj_points)

            if n_obj < 5:
                for idx in obj_indices:
                    labels[idx]['object_name'] = 'background'
                    labels[idx]['confidence']  = 0.3
                    labels[idx]['refined']     = 'dbscan_too_small'
                relabeled_count += n_obj
                continue

            min_cluster  = max(10, int(n_obj * self.min_cluster_fraction))
            min_samples  = max(3, min_cluster // 10)

            clustering  = DBSCAN(eps=eps, min_samples=min_samples).fit(obj_points)
            cluster_ids = clustering.labels_

            valid_clusters = [c for c in set(cluster_ids) if c != -1]

            if not valid_clusters:
                # All noise — relabel everything
                for idx in obj_indices:
                    labels[idx]['object_name'] = 'background'
                    labels[idx]['confidence']  = 0.2
                    labels[idx]['refined']     = 'dbscan_all_noise'
                relabeled_count += n_obj
                continue

            cluster_sizes = {c: int((cluster_ids == c).sum()) for c in valid_clusters}
            keep_clusters = {c for c, sz in cluster_sizes.items() if sz >= min_cluster}

            # Always keep at least the largest cluster
            if not keep_clusters:
                keep_clusters = {max(cluster_sizes, key=cluster_sizes.get)}

            for i, idx in enumerate(obj_indices):
                cid = cluster_ids[i]
                if cid == -1 or cid not in keep_clusters:
                    labels[idx]['object_name'] = 'background'
                    labels[idx]['confidence']  = 0.3
                    labels[idx]['refined']     = 'dbscan_isolated'
                    relabeled_count += 1

        return labels, relabeled_count

    # ------------------------------------------------------------------
    # dbscan, step 2 — Radius-based voting
    # ------------------------------------------------------------------

    def _radius_vote(self, points, labels, radius):
        """
        For each point, find all neighbors within `radius` and let them vote
        on the label. Relabels only when there is strong consensus that
        disagrees with the current label.

        Using BallTree with radius queries instead of fixed-K so that
        physically distant points never influence the vote.
        """
        tree        = BallTree(points, metric='euclidean')
        label_names = np.array([l['object_name'] for l in labels])

        relabeled_count = 0

        for i in tqdm(range(len(points)), desc="Radius voting"):
            neighbors = tree.query_radius([points[i]], r=radius)[0]
            neighbors = neighbors[neighbors != i]  # exclude self

            if len(neighbors) == 0:
                continue

            neighbor_labels = label_names[neighbors]
            unique, counts  = np.unique(neighbor_labels, return_counts=True)
            best_idx        = np.argmax(counts)
            majority_label  = unique[best_idx]
            majority_ratio  = counts[best_idx] / len(neighbors)

            current_label = labels[i]['object_name']

            if majority_ratio >= self.vote_threshold and majority_label != current_label:
                labels[i]['object_name'] = majority_label
                label_names[i]           = majority_label  # keep array in sync
                labels[i]['confidence']  = float(majority_ratio)
                labels[i]['refined']     = 'radius_voting'
                relabeled_count += 1

        return labels, relabeled_count

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_labeled_points(self, ply_path, metadata_path):
        pcd    = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return points, colors, metadata['point_labels']

    def _print_per_object_stats(self, labels, original_labels):
        print("\nPer-Object Statistics (Before → After):")
        print("-" * 70)

        original_counts = {}
        for lbl in original_labels:
            original_counts[lbl] = original_counts.get(lbl, 0) + 1

        final_counts = {}
        confidences  = {}
        for lbl in labels:
            obj = lbl['object_name']
            final_counts[obj] = final_counts.get(obj, 0) + 1
            confidences.setdefault(obj, []).append(lbl['confidence'])

        for obj in sorted(set(original_counts) | set(final_counts)):
            orig     = original_counts.get(obj, 0)
            final    = final_counts.get(obj, 0)
            avg_conf = float(np.mean(confidences.get(obj, [0])))
            change   = final - orig
            sign     = '+' if change >= 0 else ''
            print(f"  {obj:20s}: {orig:6,} → {final:6,} ({sign}{change:5,}) conf={avg_conf:.3f}")

    def _save_refined_points(self, points, colors, labels, output_path, colmap_dir=None):
        sparse_dir = output_path / "pcd_data" / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {sparse_dir}")

        pcd        = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(sparse_dir / "points3D.ply"), pcd, write_ascii=False)
        print(f"✓ points3D.ply saved")

        with open(sparse_dir / "points3D_metadata.json", 'w') as f:
            json.dump({'point_labels': labels, 'num_points': len(points), 'refined': True}, f, indent=2)
        print(f"✓ points3D_metadata.json saved")

        if colmap_dir:
            self._copy_colmap_files(colmap_dir, sparse_dir)

    def _copy_colmap_files(self, colmap_dir, dest_dir):
        import shutil
        colmap_dir    = Path(colmap_dir)
        original_sparse = colmap_dir / "sparse" / "0"

        if not original_sparse.exists():
            sparse_parent = colmap_dir / "sparse"
            if sparse_parent.exists():
                subdirs = [d for d in sparse_parent.iterdir() if d.is_dir()]
                if subdirs:
                    original_sparse = subdirs[0]

        if not original_sparse.exists():
            print(f"⚠ COLMAP sparse dir not found at {original_sparse}")
            return

        for fname in ("cameras.bin", "images.bin", "points3D.bin"):
            src = original_sparse / fname
            if src.exists():
                shutil.copy2(src, dest_dir / fname)
                print(f"✓ Copied {fname}")
            else:
                print(f"⚠ {fname} not found")


# ──────────────────────────────────────────────────────────────────────
# Standalone script
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refine point cloud labels (no point removal)"
    )
    parser.add_argument('--ply',      type=str, default="output/point_cloud/garden/labeled_point_cloud.ply")
    parser.add_argument('--metadata', type=str, default="output/point_cloud/garden/labeled_point_cloud_metadata.json")
    parser.add_argument('--output',   type=str, default="output/point_cloud/garden")
    parser.add_argument('--colmap-dir', type=str, default="output/sfm/garden")

    parser.add_argument('--method', choices=['knn', 'dbscan'], default='knn')
    parser.add_argument('--knn', type=int, default=40,
                        help='knn: neighbours that vote (default: 40)')
    parser.add_argument('--knn-threshold', type=float, default=0.9,
                        help='knn: neighbour agreement fraction required to relabel (default: 0.9)')
    parser.add_argument('--mean-spacing', type=float, default=None,
                        help='dbscan: fixed mean point spacing instead of estimating it')
    parser.add_argument('--dbscan-scale', type=float, default=3.0,
                        help='DBSCAN eps = mean_spacing × scale (default: 3.0)')
    parser.add_argument('--vote-radius-scale', type=float, default=5.0,
                        help='Vote radius = mean_spacing × scale (default: 5.0)')
    parser.add_argument('--vote-threshold', type=float, default=0.7,
                        help='Neighbor agreement fraction required to relabel (default: 0.7)')
    parser.add_argument('--min-cluster-fraction', type=float, default=0.005,
                        help='Min cluster size as fraction of object points (default: 0.005)')

    args = parser.parse_args()

    config = {
        'method':               args.method,
        'knn_neighbors':        args.knn,
        'knn_threshold':        args.knn_threshold,
        'mean_spacing':         args.mean_spacing,
        'dbscan_scale':         args.dbscan_scale,
        'vote_radius_scale':    args.vote_radius_scale,
        'vote_threshold':       args.vote_threshold,
        'min_cluster_fraction': args.min_cluster_fraction,
    }

    refiner = PointCloudRefiner(config)
    stats   = refiner.refine_labels(args.ply, args.metadata, args.output, args.colmap_dir)

    print(f"\n✓ Done! Total points: {stats['total_points']:,} (ALL KEPT) | Changed: {stats['total_changed']:,}")
