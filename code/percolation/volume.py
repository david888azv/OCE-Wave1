"""Approximate volume estimators for relaxed carbon clusters.

For a 2D-like cluster relaxed in 3D the convex hull may be degenerate
(all atoms in a plane).  We provide three robust estimators ordered by
sophistication, all in Å³:

  V_atomic       = N · (4π/3) r_C³            (purely extensive, sanity ref)
  V_inflated_hull = ConvexHull of points ± r_vdW · n̂              (3D hull
                    of the union of vdW spheres centred at each atom)
  V_obb          = oriented bounding box (3 PCA dimensions, with vdW pad)
                    ≈ V if the cluster is reasonably elongated/flat

The first is N-extensive by construction (V/N = const) and gives no
non-trivial L-scaling.  The second and third see the actual fractal
shape; they should give the interesting L-dependence.

Default vdW radius for C: 1.70 Å (Bondi 1964).
"""
from __future__ import annotations

import numpy as np

R_VDW_C = 1.70    # Å, Bondi
R_COV_C = 0.76    # Å, Pyykkö & Atsumi 2009
R_VDW_SI = 2.10   # Å, Bondi
R_COV_SI = 1.11   # Å, Pyykkö & Atsumi 2009
R_VDW_GE = 2.11   # Å, Bondi
R_COV_GE = 1.20   # Å, Pyykkö & Atsumi 2009


def volume_atomic(n_atoms: int, r: float = R_VDW_C) -> float:
    """N · (4π/3) r³ — trivially extensive."""
    return float(n_atoms * (4.0 / 3.0) * np.pi * r ** 3)


def volume_inflated_hull(positions: np.ndarray,
                          r: float = R_VDW_C,
                          n_dirs: int = 26) -> float:
    """Convex hull of a vdW-sphere union, sampled by 26 surface directions.

    For each atom, generate `n_dirs` points on a sphere of radius r around
    it; the convex hull of all these points enclos+es the cluster's vdW
    surface.  Robust to planar (z=0) clusters because the spheres provide
    out-of-plane volume.
    """
    from scipy.spatial import ConvexHull, QhullError
    # Fibonacci sphere directions — uniform on S²
    phi = np.pi * (3.0 - np.sqrt(5.0))
    idx = np.arange(n_dirs)
    z = 1.0 - 2.0 * idx / max(n_dirs - 1, 1)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    th = phi * idx
    dirs = np.stack([np.cos(th) * radius, np.sin(th) * radius, z], axis=1)

    pts = (positions[:, None, :] + r * dirs[None, :, :]).reshape(-1, 3)
    try:
        h = ConvexHull(pts)
        return float(h.volume)
    except QhullError:
        return float("nan")


def volume_obb_pad(positions: np.ndarray, r: float = R_VDW_C) -> float:
    """Oriented bounding box (PCA-aligned), padded by vdW radius along each axis.

    For a 2D cluster the third PCA axis has near-zero spread; padding by r
    gives a thickness of 2r, recovering a sensible volume.  Less tight
    than the inflated hull but always stable.
    """
    p = positions - positions.mean(axis=0, keepdims=True)
    if len(p) < 3:
        return volume_atomic(len(p), r)
    cov = np.cov(p.T)
    _, V = np.linalg.eigh(cov)
    proj = p @ V                       # cluster in PCA frame
    extents = proj.max(axis=0) - proj.min(axis=0) + 2 * r
    return float(np.prod(extents))


def planar_area_pad(positions: np.ndarray, r: float = R_VDW_C) -> tuple[float, float]:
    """For an essentially planar cluster: 2D convex-hull area in best-fit
    plane (padded by vdW), times effective thickness 2r.  Returns (area_A2,
    volume_A3)."""
    from scipy.spatial import ConvexHull, QhullError
    p = positions - positions.mean(axis=0, keepdims=True)
    if len(p) < 3:
        return 0.0, volume_atomic(len(p), r)
    cov = np.cov(p.T)
    w, V = np.linalg.eigh(cov)
    # discard the smallest-variance axis
    proj = p @ V[:, [2, 1]]   # 2D coords using the 2 largest-variance axes
    try:
        h = ConvexHull(proj)
        # Pad each hull vertex by r outward — approximate by inflating area
        # via Minkowski sum: A_pad ≈ A + r * P + π r² where P is perimeter.
        verts = proj[h.vertices]
        # close polygon
        closed = np.vstack([verts, verts[:1]])
        seg = closed[1:] - closed[:-1]
        P = float(np.sum(np.linalg.norm(seg, axis=1)))
        A = float(h.volume)   # 2D ConvexHull.volume == area
        A_pad = A + r * P + np.pi * r * r
        thickness = 2 * r
        return A_pad, A_pad * thickness
    except QhullError:
        return 0.0, float("nan")


def volume_mc_vdw(positions: np.ndarray,
                   r: float = R_VDW_C,
                   n_samples: int = 200_000,
                   seed: int = 0,
                   pad: float = 0.5) -> float:
    """Monte-Carlo integration of the volume of ⋃ᵢ B(rᵢ, r_vdW).

    Strategy: sample n_samples uniform points in a tight bounding box
    enlarged by `pad` Å, query a cKDTree for each sample, count the
    fraction whose nearest atom is within r_vdW.  V_box × fraction is
    an unbiased estimator of the union volume.

    Statistical RMS error: r ≈ √(p(1−p)/n_samples) × V_box; for n=2×10⁵
    this is well under 1% on typical clusters.
    """
    from scipy.spatial import cKDTree
    if len(positions) == 0:
        return 0.0
    lo = positions.min(axis=0) - (r + pad)
    hi = positions.max(axis=0) + (r + pad)
    box = hi - lo
    # Avoid zero-thickness bounding box (planar clusters)
    box = np.maximum(box, 2 * (r + pad))
    hi = lo + box
    rng = np.random.default_rng(seed)
    pts = rng.uniform(lo, hi, size=(n_samples, 3))
    tree = cKDTree(positions)
    d, _ = tree.query(pts, k=1, workers=-1)
    inside = float((d < r).sum()) / n_samples
    return float(inside * np.prod(box))


def volume_alpha_shape(positions: np.ndarray, alpha: float = 2.0) -> float:
    """Alpha-shape volume via Delaunay tetrahedralization.

    Keeps tetrahedra whose circumradius is < alpha (in Å).
    For an alpha → ∞ this returns the convex hull volume; for alpha
    matching the typical bond length (~1.4 Å for C) it carves out
    internal voids of a fractal cluster.

    For 3 ≤ N < 4 atoms, fall back to convex hull (Delaunay needs ≥ 4).
    For planar clusters (rank-2 point set), Delaunay raises an error and
    we fall back to inflated_hull.
    """
    from scipy.spatial import Delaunay, QhullError
    if len(positions) < 4:
        return volume_inflated_hull(positions)
    try:
        d = Delaunay(positions)
    except QhullError:
        return volume_inflated_hull(positions)
    tetra = positions[d.simplices]                    # (M, 4, 3)
    a = tetra[:, 0]
    b = tetra[:, 1]
    c = tetra[:, 2]
    e = tetra[:, 3]
    # tetrahedron volume = |((b-a)·((c-a)×(e-a)))| / 6
    v_tet = np.abs(np.einsum("ij,ij->i",
                              b - a,
                              np.cross(c - a, e - a))) / 6.0
    # circumradius of each tetrahedron
    # Use formula via the determinant; or compute the centre by solving
    # 3 equations: |x - a|² = |x - b|² = |x - c|² = |x - e|².
    # Vectorise: build matrix (b-a; c-a; e-a) and rhs ½(|b|²-|a|²; |c|²-|a|²; |e|²-|a|²)
    M = np.stack([b - a, c - a, e - a], axis=1)        # (n, 3, 3)
    rhs = 0.5 * np.stack([
        (b * b).sum(-1) - (a * a).sum(-1),
        (c * c).sum(-1) - (a * a).sum(-1),
        (e * e).sum(-1) - (a * a).sum(-1),
    ], axis=1)
    try:
        centres = np.linalg.solve(M, rhs[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        return volume_inflated_hull(positions)
    R = np.linalg.norm(centres - a, axis=1)
    keep = R < alpha
    return float(v_tet[keep].sum())


if __name__ == "__main__":
    import json
    from pathlib import Path
    cache = Path(__file__).resolve().parent / "data" / "phase2_smoke" / "clusters_relax.json"
    if cache.exists():
        records = json.loads(cache.read_text())
        print(" N    V_atomic   V_obb      V_hull(vdW)  V_planar  V_mc(vdW)  V_alpha=2.0")
        for r in records:
            pos = np.array(r["positions_relaxed"])
            v_at = volume_atomic(r["n_atoms"])
            v_obb = volume_obb_pad(pos)
            v_hull = volume_inflated_hull(pos)
            _, v_pl = planar_area_pad(pos)
            v_mc = volume_mc_vdw(pos, n_samples=200_000)
            v_a = volume_alpha_shape(pos, alpha=2.0)
            print(f" {r['n_atoms']:3d}   {v_at:8.1f}  {v_obb:8.1f}  "
                  f"{v_hull:8.1f}    {v_pl:7.1f}  {v_mc:8.1f}  {v_a:8.1f}")
