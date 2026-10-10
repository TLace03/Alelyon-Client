//! A bounding-volume hierarchy over one mesh's triangles, built on the host.
//!
//! The build is a binned surface-area-heuristic (SAH) split, deterministic for
//! a given mesh: no randomness, no hashing of addresses, and every tie broken
//! by the lowest axis and then the lowest bin. Meshes are rigid in v0, so each
//! is built once, when the scene is loaded, and instanced by pose; a deformable
//! mesh will need a device-side refit or rebuild (the design note's phase 3).
//!
//! Layout (mirrored by `Node` in `kernels/common.glsl`): node 0 is the root; an
//! interior node's children are the pair `left, left + 1`; a leaf names a run
//! of `count` triangles starting at `first` in [`Bvh::order`].

use crate::math::{Vec3, max3, min3, sub};
use crate::mesh::Mesh;

/// The most triangles a leaf may hold.
pub const MAX_LEAF: usize = 4;

/// The deepest tree the kernels' traversal stack can walk
/// (`STACK_DEPTH` in `kernels/common.glsl`).
pub const MAX_DEPTH: u32 = 32;

const BINS: usize = 16;

/// One node: its box and either its children or its triangles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvhNode {
    /// The box's minimum corner.
    pub bmin: Vec3,
    /// The box's maximum corner.
    pub bmax: Vec3,
    /// The left child (interior) or the first triangle in `order` (leaf).
    pub left_or_first: u32,
    /// The number of triangles; 0 for an interior node.
    pub count: u32,
}

/// A built hierarchy.
#[derive(Clone, Debug, PartialEq)]
pub struct Bvh {
    /// The nodes, root first.
    pub nodes: Vec<BvhNode>,
    /// The mesh's triangle indices in leaf order.
    pub order: Vec<u32>,
    /// The depth of the deepest leaf (the root alone has depth 1).
    pub depth: u32,
}

#[derive(Clone, Copy)]
struct Aabb {
    lo: Vec3,
    hi: Vec3,
}

impl Aabb {
    const EMPTY: Aabb = Aabb {
        lo: [f32::INFINITY; 3],
        hi: [f32::NEG_INFINITY; 3],
    };

    fn grow(&mut self, other: &Aabb) {
        self.lo = min3(self.lo, other.lo);
        self.hi = max3(self.hi, other.hi);
    }

    fn grow_point(&mut self, p: Vec3) {
        self.lo = min3(self.lo, p);
        self.hi = max3(self.hi, p);
    }

    fn area(&self) -> f32 {
        if self.lo[0] > self.hi[0] {
            return 0.0;
        }
        let d = sub(self.hi, self.lo);
        2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
    }
}

/// Build the hierarchy of `mesh` with leaves of at most [`MAX_LEAF`]
/// triangles. Refuses (returns `Err`) an empty mesh or a tree deeper than
/// [`MAX_DEPTH`].
pub fn build(mesh: &Mesh) -> Result<Bvh, String> {
    build_with_leaf(mesh, MAX_LEAF)
}

/// [`build`] with leaves of at most `max_leaf` (at least 1) triangles: a
/// measurement lever; the kernels read each leaf's count, so any size works.
pub fn build_with_leaf(mesh: &Mesh, max_leaf: usize) -> Result<Bvh, String> {
    if max_leaf == 0 {
        return Err("a leaf holds at least one triangle".into());
    }
    let n = mesh.triangles.len();
    if n == 0 {
        return Err("a mesh with no triangles has no hierarchy".into());
    }
    let boxes: Vec<Aabb> = mesh
        .triangles
        .iter()
        .map(|t| {
            let mut b = Aabb::EMPTY;
            for &k in t {
                b.grow_point(mesh.positions[k as usize]);
            }
            b
        })
        .collect();
    let centroids: Vec<Vec3> = boxes
        .iter()
        .map(|b| {
            [
                0.5 * (b.lo[0] + b.hi[0]),
                0.5 * (b.lo[1] + b.hi[1]),
                0.5 * (b.lo[2] + b.hi[2]),
            ]
        })
        .collect();
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut nodes = vec![BvhNode {
        bmin: [0.0; 3],
        bmax: [0.0; 3],
        left_or_first: 0,
        count: 0,
    }];
    // (node index, start, end, depth)
    let mut work = vec![(0usize, 0usize, n, 1u32)];
    let mut depth = 1;
    while let Some((node, start, end, d)) = work.pop() {
        depth = depth.max(d);
        let mut bounds = Aabb::EMPTY;
        let mut cbounds = Aabb::EMPTY;
        for &t in &order[start..end] {
            bounds.grow(&boxes[t as usize]);
            cbounds.grow_point(centroids[t as usize]);
        }
        nodes[node].bmin = bounds.lo;
        nodes[node].bmax = bounds.hi;
        let count = end - start;
        let split = if count <= 1 {
            None
        } else {
            best_split(
                &order[start..end],
                &boxes,
                &centroids,
                &cbounds,
                bounds.area(),
                max_leaf,
            )
        };
        let mut mid = split.map(|(axis, bin)| {
            let lo = cbounds.lo[axis];
            let extent = cbounds.hi[axis] - lo;
            let slice = &mut order[start..end];
            // stable partition: the order inside each side is the old order
            let (mut left, mut right): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
            for &t in slice.iter() {
                if bin_of(centroids[t as usize][axis], lo, extent) <= bin {
                    left.push(t);
                } else {
                    right.push(t);
                }
            }
            let m = left.len();
            slice[..m].copy_from_slice(&left);
            slice[m..].copy_from_slice(&right);
            start + m
        });
        if mid.is_none_or(|m| m == start || m == end) {
            // no SAH split separates the triangles; a node too large to be a
            // leaf (every centroid identical) is halved by index instead
            mid = (count > max_leaf).then_some(start + count / 2);
        }
        match mid {
            Some(m) => {
                let left = nodes.len();
                nodes.push(nodes[node]);
                nodes.push(nodes[node]);
                nodes[node].left_or_first = left as u32;
                nodes[node].count = 0;
                // right first on the stack, so the left subtree is built (and
                // numbered) first: the layout is the same on every run
                work.push((left + 1, m, end, d + 1));
                work.push((left, start, m, d + 1));
            }
            None => {
                nodes[node].left_or_first = start as u32;
                nodes[node].count = count as u32;
            }
        }
    }
    if depth > MAX_DEPTH {
        return Err(format!(
            "the hierarchy is {depth} levels deep and the traversal stack holds {MAX_DEPTH}"
        ));
    }
    Ok(Bvh {
        nodes,
        order,
        depth,
    })
}

fn bin_of(c: f32, lo: f32, extent: f32) -> usize {
    let b = ((c - lo) / extent * BINS as f32) as isize;
    b.clamp(0, BINS as isize - 1) as usize
}

/// The SAH-best (axis, last left bin), or `None` when no split beats a leaf
/// and the node is small enough to be one.
#[allow(clippy::needless_range_loop)] // `axis` indexes three parallel arrays
fn best_split(
    tris: &[u32],
    boxes: &[Aabb],
    centroids: &[Vec3],
    cbounds: &Aabb,
    parent_area: f32,
    max_leaf: usize,
) -> Option<(usize, usize)> {
    let count = tris.len();
    let leaf_cost = count as f32;
    let mut best: Option<(f32, usize, usize)> = None;
    for axis in 0..3 {
        let lo = cbounds.lo[axis];
        let extent = cbounds.hi[axis] - lo;
        if extent <= 0.0 {
            continue;
        }
        let mut bin_box = [Aabb::EMPTY; BINS];
        let mut bin_n = [0usize; BINS];
        for &t in tris {
            let b = bin_of(centroids[t as usize][axis], lo, extent);
            bin_box[b].grow(&boxes[t as usize]);
            bin_n[b] += 1;
        }
        // right-to-left sweep of areas and counts
        let mut right_area = [0.0f32; BINS];
        let mut right_n = [0usize; BINS];
        let mut acc = Aabb::EMPTY;
        let mut accn = 0;
        for b in (1..BINS).rev() {
            acc.grow(&bin_box[b]);
            accn += bin_n[b];
            right_area[b] = acc.area();
            right_n[b] = accn;
        }
        let mut left = Aabb::EMPTY;
        let mut leftn = 0;
        for b in 0..BINS - 1 {
            left.grow(&bin_box[b]);
            leftn += bin_n[b];
            let rn = right_n[b + 1];
            if leftn == 0 || rn == 0 {
                continue;
            }
            let cost = 1.0
                + (left.area() * leftn as f32 + right_area[b + 1] * rn as f32)
                    / parent_area.max(f32::MIN_POSITIVE);
            if best.is_none_or(|(c, _, _)| cost < c) {
                best = Some((cost, axis, b));
            }
        }
    }
    match best {
        Some((cost, axis, bin)) if cost < leaf_cost || count > max_leaf => Some((axis, bin)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_invariants(mesh: &Mesh, bvh: &Bvh, max_leaf: usize) {
        let n = mesh.triangles.len();
        let mut seen = vec![0u32; n];
        let contains = |outer: &BvhNode, lo: Vec3, hi: Vec3| {
            (0..3).all(|i| outer.bmin[i] <= lo[i] && hi[i] <= outer.bmax[i])
        };
        for node in &bvh.nodes {
            if node.count > 0 {
                assert!(node.count as usize <= max_leaf);
                for k in node.left_or_first..node.left_or_first + node.count {
                    let t = bvh.order[k as usize] as usize;
                    seen[t] += 1;
                    for &v in &mesh.triangles[t] {
                        let p = mesh.positions[v as usize];
                        assert!(contains(node, p, p));
                    }
                }
            } else {
                for c in [node.left_or_first, node.left_or_first + 1] {
                    let child = &bvh.nodes[c as usize];
                    assert!(contains(node, child.bmin, child.bmax));
                }
            }
        }
        assert!(
            seen.iter().all(|&s| s == 1),
            "every triangle in exactly one leaf"
        );
        assert!(bvh.depth <= MAX_DEPTH);
    }

    #[test]
    fn every_triangle_is_in_one_leaf_and_every_box_contains_its_contents() {
        for mesh in [
            Mesh::cuboid([0.5, 0.2, 0.1]),
            Mesh::sphere(0.3, 32, 16),
            Mesh::cylinder(0.05, 0.1, 48),
            Mesh::plane(5.0, 5.0, 16, 16),
        ] {
            let bvh = build(&mesh).unwrap();
            check_invariants(&mesh, &bvh, MAX_LEAF);
        }
    }

    #[test]
    fn every_leaf_size_keeps_the_invariants() {
        let mesh = Mesh::sphere(0.3, 32, 16);
        for max_leaf in [1, 2, 8, 16] {
            let bvh = build_with_leaf(&mesh, max_leaf).unwrap();
            check_invariants(&mesh, &bvh, max_leaf);
        }
        assert!(build_with_leaf(&mesh, 0).is_err());
    }

    #[test]
    fn the_build_is_deterministic() {
        let mesh = Mesh::sphere(0.3, 32, 16);
        assert_eq!(build(&mesh).unwrap(), build(&mesh).unwrap());
    }

    #[test]
    fn coincident_triangles_still_split_into_small_leaves() {
        let mut mesh = Mesh::default();
        for _ in 0..40 {
            let base = mesh.positions.len() as u32;
            mesh.positions
                .extend([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
            mesh.normals.extend([[0.0, 0.0, 1.0]; 3]);
            mesh.triangles.push([base, base + 1, base + 2]);
        }
        let bvh = build(&mesh).unwrap();
        check_invariants(&mesh, &bvh, MAX_LEAF);
    }

    #[test]
    fn an_empty_mesh_is_refused() {
        assert!(build(&Mesh::default()).is_err());
    }
}
