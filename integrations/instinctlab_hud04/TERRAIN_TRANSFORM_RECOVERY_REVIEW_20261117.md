# Terrain transform / evidence recovery review — 2026-11-17

## Scope and recovered evidence

- Delivery base: `fd12fe13ce9293903d04e83d974987025f03b75e`. At recovery time the branch HEAD matched this exact commit. All 85 tracked blobs were downloaded into a new directory; the previous local worktree was absent. No reset or old startup/publication script was run.
- Pinned InstinctLab source: `de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057`.
- Candidate IsaacLab source: tag `v2.3.2`, recursive tree SHA `37ddf626871758333d6ed89cf64ad702aef127d0`. This is the candidate integration version, NOT evidence about the unavailable original NPZ exporter or completed GPU task runtime.
- Re-executed the unchanged baseline integration suite: **123 tests passed**. After the native-rate fix and one new regression test: **124 passed**. CPU software tests only.
- The two previously completed GPU tasks remain on the do-not-resubmit list. This stage makes no GPU service request or submission.

## 1. Exact crop contract, not an outside-vertex approximation

[InstinctLab mesh_terrains.py, lines 20–78](https://github.com/project-instinct/InstinctLab/blob/de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057/source/instinctlab/instinctlab/terrains/trimesh/mesh_terrains.py#L20-L78)
loads the mesh with `trimesh.load(..., force="mesh")`, then crops it before estimating border height.

[crop_terrain_mesh_aabb](https://github.com/project-instinct/InstinctLab/blob/de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057/source/instinctlab/instinctlab/terrains/trimesh/utils.py#L9-L81)
sequentially calls `trimesh.intersections.slice_mesh_plane` in this order:

1. x <= +size_x/2;
2. x >= -size_x/2;
3. y <= +size_y/2;
4. y >= -size_y/2.

It does not request caps, and the caller supplies no z crop. The generic perceptive configuration uses size `(9, 12)`, hence conditional raw-coordinate bounds x ±4.5 and y ±6. These are not accepted HUD04 terrain dimensions or verified source units.

The prior counts (59 triangles with at least one outside vertex, 33 wholly outside a common plane) are NOT clipped triangle counts. Intersecting triangles must be split, not simply dropped. Generated slice vertices and mesh vertex merging affect the subsequent border-vertex population. Exact reproduction must document the actual trimesh version and load/processing semantics; triangle-soup vertex multiplicity cannot silently replace mesh vertices in the border mean.

**Not executed on the real STL this turn:** clipping, final face counts, border population, border height. The raw STL is not in the repository or current recovered filesystem; trimesh is also not installed in this CPU sandbox.

## 2. Border mean and shared translation

On the cropped mesh, the border mask is the strict OR condition

```text
abs(x) > size_x/2 - 0.05 OR abs(y) > size_y/2 - 0.05
```

The border height b is the arithmetic mean of those vertices' z coordinates, not minimum, median, area-weighted mean, or a fitted ground plane. NaN falls back to zero with a warning. For `(9,12)`, the raw thresholds are 4.45 and 5.95. The code assumes a flat border; it does not validate flatness. A future audit must report border z range/spread and the fallback explicitly, rather than treating a fallback as a calibrated surface.

The terrain function translates both mesh and returned origin by

```text
u = (size_x/2, size_y/2, -b).
```

[IsaacLab v2.3.2 terrain_generator.py](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/terrains/terrain_generator.py)
then applies the same translations to mesh and origin:

- `_get_terrain_mesh`: subtract `(size_x/2, size_y/2, 0)`;
- `_add_sub_terrain`: add `((row+0.5)*size_x, (col+0.5)*size_y, 0)`;
- final layout centering: subtract `(num_rows*size_x/2, num_cols*size_y/2, 0)`.

Thus, for this candidate generator and a noncached equivalent mesh, the final translation is

```text
t = ((row+0.5-num_rows/2)*size_x,
     (col+0.5-num_cols/2)*size_y,
     -b).
mesh_world = crop(mesh_raw) + t
```

[TerrainMotion.match_scene / _get_motion_based_origin](https://github.com/project-instinct/InstinctLab/blob/de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057/source/instinctlab/instinctlab/motion_reference/motion_files/terrain_motion.py#L64-L155)
stores and samples the generator terrain origins. Its inherited
[AmassMotion initialization and reference fill](https://github.com/project-instinct/InstinctLab/blob/de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057/source/instinctlab/instinctlab/motion_reference/motion_files/amass_motion.py#L193-L245)
adds that origin to the base; `fill_motion_data` also adds it to world base/link positions (lines 396–402).
The actual G1 perceptive config disables below-ground lifting and sets starting-height offset to zero.

**Conditional consequence:** if the NPZ and raw terrain already share units/frame, correctly applying this same t to both preserves relative geometry on retained surfaces. Border translation alone cannot repair a source/terrain misalignment. Translating only mesh or only robot would introduce a false gap. Clipping can change the queried surface and must still be recomputed. Cache contents and the actual GPU runtime have not been recovered or validated.

## 3. Lateral contacts and support windows remain unresolved

- `MotionMatchedTerrainCfg` does not inherit `WallTerrainCfgMixin`. The `generate_wall` wrapper returns without adding perimeter walls when `wall_prob` is absent. The default motion-matched task therefore does not acquire random perimeter walls merely because the function is decorated.
- Vertical faces already inside the supplied terrain mesh remain relevant. They are distinct from optional generated perimeter walls.
- A world-down support extremum per capsule plus an XY upper-envelope ray cannot resolve lateral palm/knee contact, undercuts or all obstacle penetrations. A near-zero or absent vertical gap is not a contact label or a signed collision distance.
- Before labeling support, retain face identity/normals and evaluate appropriate directional collider support against candidate surfaces, bounded 3D proximity/collision information, and fixed material-point temporal motion. A posture-dependent world-down extremum is not a fixed material point and cannot supply slip velocity by finite difference.
- Keep all four root/quaternion hypotheses unselected. A hypothesis with fewer below-envelope points is not proven correct.
- Existing windows 611–641, 659–689 and 711–741 are diagnostic windows, NOT accepted hand/foot/knee support intervals. No new support schedule was generated here.

## 4. Native-rate defect fixed without changing source data

The previous HUD04 overlay selected `motion_interpolate_bilinear` but left `AmassMotionCfg.motion_target_framerate` at its upstream default **50.0**. `_pack_retargetted_motion_sequence` invokes that function and then changes its stored framerate to the target. This conflicts with keeping the original 90 Hz reference samples.

The local continuation changes only the HUD04 loader option to `motion_interpolate_func=None, skip_frames=0` and removes the unused import. No source NPZ, model, reference-manifest acceptance rule, original probe, G1 configuration or control/physics rate is changed. `clip_joint_ref_to_robot_limits=False` remains in force.

The regression executes the real `_motion_cfg` function body with explicit CPU configuration stubs and asserts native sampling options, no clipping/retargeting and no automatic height correction. It is not a real IsaacLab config construction or accepted motion test. Future 50 Hz policy-time lookups may still sample a 90 Hz reference; preserving its stored samples does not mean the policy runs at 90 Hz or prove tracking quality.

## 5. Missing evidence and next gate

Not recovered in this sandbox:

- Original `/workspace/HUD04_RECOVERY_STATUS.md`, old worktree and its four uncommitted changes. The published commit was recovered instead, not represented as restoration of those local bytes.
- The four original NPZ files, two YAMLs and terrain STL, plus detailed conditional-point reports and independent verifier outputs. Their hashes/summaries are available in the published reviews; the 6696 real points were NOT recomputed this turn.
- The exact NPZ exporter/code revision, generating URDF, named root frame, quaternion convention and world/terrain units/alignment, and original human acceptance records. Loader conventions are not exporter provenance.
- Live Flux server/task state and detailed GPU evidence; no live check was necessary for this CPU-only phase. A live status/duplicate check remains mandatory before any future GPU submission.
- W&B project access has not been retested and the prior invalid-audience problem is not assumed fixed.

Next: recover the original Drive dataset read-only and verify each file hash against the published records; separately seek the exporter/provenance records (not inferred from numeric fit). Then reproduce exact processed terrain surfaces, review lateral and hand/foot/knee support candidates under explicit coordinate hypotheses, and only then plan bounded retargeting with process supervision. The previously recorded solver limits are planning records, not implemented enforcement or authorization to start.

**No reference accepted; no solver called; no GPU submitted; no PPO run; training_ready=false.**