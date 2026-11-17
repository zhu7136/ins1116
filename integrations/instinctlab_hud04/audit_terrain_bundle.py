"""Read-only TerrainMotion bundle diagnostics; never acceptance or simulation.

Mirrors static requirements of pinned InstinctLab, without importing its GPU stack.
Crop diagnostics assume the supplied raw coordinates share the loader's units/frame;
that assumption is NOT verified here. No clipping, data conversion or mesh mutation.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('terrain_candidate_audit', ROOT / 'audit_motion.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


class UniqueSafeLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result:
            raise ValueError('Metadata requires unique string mapping keys')
        result[key] = loader.construct_object(value_node, deep=True)
    return result


UniqueSafeLoader.add_constructor('tag:yaml.org,2002:map', unique_mapping)


def mesh_crop_diagnostic(path, tile_size):
    result = A.audit_binary_stl(path)
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    triangles = np.frombuffer(path.read_bytes(), dtype=dtype, offset=84)['vertices']
    half = np.asarray(tile_size, dtype=float) / 2
    xy = triangles[:, :, :2]
    outside = (np.abs(xy) > half).any(axis=2)
    # All triangle vertices beyond one common crop plane implies it is removed.
    removed = ((xy > half).all(axis=1) | (xy < -half).all(axis=1)).any(axis=1)
    result.update(vertex_occurrences_outside_raw_crop=int(outside.sum()),
        triangles_touching_outside_raw_crop=int(outside.any(axis=1).sum()),
        triangles_entirely_beyond_one_raw_crop_plane=int(removed.sum()),
        border_height_offset_reproduced=False, exact_clipped_mesh_computed=False)
    return result


def audit_bundle(dataset, metadata_name, urdf, tile_size):
    dataset = Path(dataset).resolve()
    tile_size = tuple(float(x) for x in tile_size)
    if len(tile_size) != 2 or not all(math.isfinite(x) and x > 0 for x in tile_size):
        raise ValueError('Tile size must contain two finite positive values')
    metadata_path = A.contained(dataset, metadata_name)
    if metadata_path.stat().st_size > 65536:
        raise ValueError('Metadata size limit')
    text = metadata_path.read_text()
    data = yaml.load(text, Loader=UniqueSafeLoader)
    # Reuse bounded path, type, reference and unique-terrain-ID checks.
    links = A.audit_metadata(metadata_path, dataset)
    names, model = A.C.validate_model(urdf)
    terrain_ids = [row['terrain_id'] for row in data['terrains']]
    issues = []
    if terrain_ids != list(range(len(terrain_ids))):
        issues.append('terrain_ids_must_equal_list_indices_for_pinned_match_scene')
    if not links['file_and_id_links_resolve']:
        issues.append('missing_file_or_unmatched_terrain_id')
    weights = []
    for row in data['motion_files']:
        value = row.get('weight', 1.0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            issues.append('motion_weights_must_be_finite_nonnegative_numbers')
            weights.append(None)
        else:
            weights.append(float(value))
    if all(w is not None for w in weights) and not any(w > 0 for w in weights):
        issues.append('motion_weights_have_no_positive_mass')
    motions, terrains, errors = [], [], []
    for row in data['motion_files']:
        path = A.contained(dataset, row['motion_file'])
        if not path.is_file():
            continue
        try:
            motion = A.audit_npz(path, names, model)
            with np.load(path, allow_pickle=False) as arrays:
                xy = arrays['base_pos_w'][:, :2].astype(float)
            outside = (np.abs(xy) > np.array(tile_size) / 2).any(axis=1)
            motion.update(terrain_id=row['terrain_id'],
                root_frames_outside_raw_crop=int(outside.sum()),
                root_xy_raw_bounds=[xy.min(axis=0).tolist(), xy.max(axis=0).tolist()])
            motions.append(motion)
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
            errors.append(dict(file=row['motion_file'], error=str(exc)))
    for row in data['terrains']:
        path = A.contained(dataset, row['terrain_file'])
        if not path.is_file():
            continue
        try:
            terrain = mesh_crop_diagnostic(path, tile_size)
            terrain['terrain_id'] = row['terrain_id']
            terrains.append(terrain)
        except (ValueError, OSError) as exc:
            errors.append(dict(file=row['terrain_file'], error=str(exc)))
    return dict(schema_version=1, mode='read_only_terrain_candidate_diagnostic',
        instinctlab_commit=A.C.INSTINCTLAB_COMMIT, model_sha256=A.C.sha256(urdf),
        metadata=links, terrain_ids_in_list_order=terrain_ids, sampling_weights=weights,
        tile_size_raw_units=list(tile_size), loader_contract_issues=sorted(set(issues)),
        static_loader_contract_pass=not issues and not errors,
        motions=motions, terrains=terrains, errors=errors,
        all_referenced_motion_numerics_pass=bool(motions) and
            len(motions) == len(data['motion_files']) and all(m['numerical_limits_pass'] for m in motions),
        crop_diagnostics_assume_common_raw_frame_and_units=True,
        terrain_frame_or_units_verified=False, root_frame_verified=False,
        contact_policy='not_configured_or_accepted', physical_replay_verified=False,
        source_data_modified=False, accepted_reference=False, training_ready=False,
        gpu_submissions=0, solver_calls=0, ppo_updates=0,
        limitations=['Static loader compatibility is not reference or terrain acceptance',
            'Crop counts concern raw-coordinate vertices/root positions, not feet/hands or valid contacts',
            'No exact crop, border-height shift, generator/wall transform or scene-origin simulation',
            'Do not insert climbing data into the feet-only Plane task'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--metadata', required=True, help='Relative YAML path inside dataset')
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--tile-size', nargs=2, type=float, required=True, metavar=('X', 'Y'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = audit_bundle(args.dataset, args.metadata, args.urdf, args.tile_size)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.report:
        with args.report.open('x') as stream:
            stream.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()