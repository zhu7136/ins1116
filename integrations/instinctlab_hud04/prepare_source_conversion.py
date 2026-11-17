"""Lossless, non-training source staging; NOT G1-to-HUD04 retargeting.

Only pinned four original motions are supported. No coordinate inference, clipping,
resampling, optimization, simulation, network access or acceptance-manifest output.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('source_conversion_audit', HERE / 'audit_motion.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
MODEL_SHA256 = '865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_arrays(source, target_names, limits):
    names = source['joint_names'].tolist()
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(target_names):
        raise ValueError('Expected 29 unique matching BODY names; no head action')
    if any('head' in n for n in names):
        raise ValueError('Head must not enter action channels')
    order = np.array([names.index(n) for n in target_names], dtype=np.int64)
    inverse = np.argsort(order)
    q = source['joint_pos'][:, order].copy()
    p, quat = source['base_pos_w'], source['base_quat_w']
    fps = float(source['framerate'].item())
    if fps != 90 or q.ndim != 2 or q.shape[1] != 29 or len(q) < 3:
        raise ValueError('Expected native 90 Hz, >=3 samples, 29 body channels')
    if p.shape != (len(q), 3) or quat.shape != (len(q), 4):
        raise ValueError('Invalid raw root shapes')
    if not all(np.isfinite(v).all() for v in (q, p, quat)):
        raise ValueError('Nonfinite source')
    lo, hi, vmax = (np.array([limits[n][k] for n in target_names])
                    for k in ('lower', 'upper', 'velocity'))
    q64 = q.astype(np.float64)
    position_bad = (q64 < lo-1e-6) | (q64 > hi+1e-6)
    speed_bad = np.abs(np.diff(q64, axis=0))*fps > vmax+1e-6
    arrays = dict(
        source_frame_index=np.arange(len(q), dtype=np.int64),
        time_seconds=np.arange(len(q), dtype=np.float64)/fps,
        source_framerate=source['framerate'].copy(),
        original_source_joint_names=source['joint_names'].copy(),
        target_body_name_order=np.asarray(target_names, dtype=source['joint_names'].dtype),
        source_columns_in_target_name_order=order,
        inverse_column_permutation=inverse,
        source_joint_values_name_reordered_only=q,
        unverified_raw_root_position=p.copy(),
        unverified_raw_root_quaternion=quat.copy(),
        source_vs_target_position_violation_mask=position_bad,
        source_vs_target_speed_violation_mask=speed_bad,
    )
    # Do not expose the standard consumer fields (joint_pos/base_pos_w/etc.).
    # The suffix and schema deliberately cannot be used as an accepted reference.
    assert np.array_equal(q[:, inverse], source['joint_pos'])
    return arrays


def verify_arrays(staged, source, expected):
    if set(staged) != set(expected):
        raise ValueError('Staging keys changed')
    for k, value in expected.items():
        if staged[k].dtype != value.dtype or not np.array_equal(staged[k], value):
            raise ValueError('Staging readback mismatch: '+k)
    recovered = staged['source_joint_values_name_reordered_only'][:, staged['inverse_column_permutation']]
    if recovered.dtype != source['joint_pos'].dtype or not np.array_equal(recovered, source['joint_pos']):
        raise ValueError('Lossless inverse-column check failed')
    for original, key in [('base_pos_w', 'unverified_raw_root_position'),
                          ('base_quat_w', 'unverified_raw_root_quaternion'),
                          ('framerate', 'source_framerate'),
                          ('joint_names', 'original_source_joint_names')]:
        if staged[key].dtype != source[original].dtype or not np.array_equal(staged[key], source[original]):
            raise ValueError('Original value/dtype not preserved: '+original)


def prepare(dataset, urdf, output):
    dataset, urdf, output = (Path(p).resolve() for p in (dataset, urdf, output))
    if output.exists() or output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError('Output must be a NEW directory outside the original dataset')
    review = json.loads((HERE/'MOTION_REVIEW_20261117.json').read_text())
    originals = review['motions']+review['metadata']+[review['terrain']]
    for row in originals:
        if sha(A.contained(dataset, row['name'])) != row['sha256']:
            raise ValueError('Original input hash mismatch: '+row['name'])
    if sha(urdf) != MODEL_SHA256:
        raise ValueError('Reviewed target model changed')
    target_names, limits = A.C.validate_model(urdf)
    output.mkdir(parents=True, exist_ok=False)
    rows=[]
    for item in review['motions']:
        src = dataset/item['name']
        audit = A.audit_npz(src, target_names, limits)
        with np.load(src, allow_pickle=False) as data:
            source = {key:data[key].copy() for key in data.files}
        arrays = stage_arrays(source, target_names, limits)
        dest = output/(src.stem+'.source-stage.npz')
        with dest.open('xb') as f:
            np.savez_compressed(f, **arrays)
        with np.load(dest, allow_pickle=False) as data:
            verify_arrays(data, source, arrays)
        assert audit['position_violating_frames'] == int(arrays['source_vs_target_position_violation_mask'].any(axis=1).sum())
        assert audit['overspeed_intervals'] == int(arrays['source_vs_target_speed_violation_mask'].any(axis=1).sum())
        rows.append(dict(source_name=src.name, source_sha256=sha(src), staged_name=dest.name,
            staged_sha256=sha(dest), frames=len(source['joint_pos']), fps=90,
            original_joint_dtype=str(source['joint_pos'].dtype), lossless_roundtrip_verified=True,
            position_violating_frames=audit['position_violating_frames'], overspeed_intervals=audit['overspeed_intervals'],
            body_channels=29, head_action_channels=0, root_and_quaternion_interpretation='unverified_raw_values'))
    for row in originals:
        if sha(dataset/row['name']) != row['sha256']:
            raise ValueError('Original input changed during staging')
    report=dict(schema_version=1, status='source_staging_only_NOT_HUD04_reference', files=rows,
        target_model_sha256=MODEL_SHA256, total_frames=sum(x['frames'] for x in rows),
        original_input_hashes_verified=originals, original_input_file_count=len(originals),
        head_lock=dict(head_yaw_joint=0, head_pitch_joint=0, included_in_actions=False),
        transformations_applied=['lossless_joint_column_permutation_by_name', 'derived_timestamps_and_violation_masks'],
        joint_sign_changes=False, root_transform_applied=False, quaternion_reordered=False,
        terrain_scaled=False, joint_clipping=False, resampling=False, original_data_modified=False,
        source_exporter_proven=False, generating_model_proven=False, coordinate_hypothesis_selected=False,
        accepted_reference=False, training_ready=False, gpu_submissions=0, solver_calls=0, ppo_updates=0,
        blockers=['Bind the actual Step1 exporter revision/command and generating URDF',
                  'Confirm root frame, quaternion order and world/STL units/alignment',
                  'Review material-point hand/foot/knee support constraints before a bounded solver'])
    manifest=output/'STAGING_MANIFEST.json'
    with manifest.open('x') as f:
        json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
    assert json.loads(manifest.read_text()) == report
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.dataset,args.urdf,args.output),indent=2,ensure_ascii=False))