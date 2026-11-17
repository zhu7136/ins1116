"""Explicit allowlist; original + lossless staging backup, NOT accepted motion."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
RAW=Path('/workspace/hud04-original-drive-20261117')
STAGED=Path('/workspace/hud04-source-staging-20261117')
BASE=Path('/workspace/hud04-recovery-bf4a057/integrations/instinctlab_hud04')
SEARCH=Path('/workspace/project-instinct-data-search-20261117')
RUNTIME=Path('/workspace/hud04-runtime-preflight-20261117')


def digest(raw):return hashlib.sha256(raw).hexdigest()


def main():
    manifest=json.loads((STAGED/'STAGING_MANIFEST.json').read_text())
    assert manifest['accepted_reference'] is False and manifest['training_ready'] is False
    files={}
    for row in manifest['original_input_hashes_verified']:
        name=row['name'];assert Path(name).name==name
        raw=(RAW/name).read_bytes();assert digest(raw)==row['sha256']
        files['original/'+name]=raw
    counts=[]
    for row in manifest['files']:
        raw=(STAGED/row['staged_name']).read_bytes();assert digest(raw)==row['staged_sha256']
        files['source_staging/'+row['staged_name']]=raw
        with np.load(RAW/row['source_name'],allow_pickle=False) as a,np.load(STAGED/row['staged_name'],allow_pickle=False) as b:
            q=a['joint_pos'];restored=b['source_joint_values_name_reordered_only'][:,b['inverse_column_permutation']]
            assert restored.dtype==q.dtype and np.array_equal(restored,q)
            assert np.array_equal(b['unverified_raw_root_position'],a['base_pos_w'])
            assert np.array_equal(b['unverified_raw_root_quaternion'],a['base_quat_w'])
            assert float(a['framerate'].item())==90 and len(q)==row['frames']
            assert len(a['joint_names'])==29 and not any('head' in n for n in a['joint_names'].tolist())
            assert int(np.any(b['source_vs_target_position_violation_mask'],axis=1).sum())==row['position_violating_frames']
            assert int(np.any(b['source_vs_target_speed_violation_mask'],axis=1).sum())==row['overspeed_intervals']
            counts.append(dict(name=row['source_name'],frames=len(q),fps=90,lossless_roundtrip=True,
                position_violating_frames=row['position_violating_frames'],overspeed_intervals=row['overspeed_intervals']))
    files['source_staging/STAGING_MANIFEST.json']=(STAGED/'STAGING_MANIFEST.json').read_bytes()
    evidence={
        'source_interface_chain.json':SEARCH/'source_interface_chain.json',
        'motion_source_comparison.json':SEARCH/'motion_source_comparison.json',
        'SEARCH_FINDINGS_20261117.md':SEARCH/'SEARCH_FINDINGS_20261117.md',
        'PROCESSED_TERRAIN_REVIEW_20261117.md':BASE/'PROCESSED_TERRAIN_REVIEW_20261117.md',
        'SOURCE_CONVERSION_PREPARATION_20261117.md':BASE/'SOURCE_CONVERSION_PREPARATION_20261117.md',
        'SOURCE_ROOT_TARGET_JACOBIAN_REVIEW_20261117.json':RUNTIME/'SOURCE_ROOT_TARGET_JACOBIAN_REVIEW_20261117.json',
        'MULTIPATCH_REVIEW_zh_20261117.md':Path('/workspace/hud04-multipatch-review-20261117/REVIEW_zh_20261117.md'),
        'CONTACT_CORRESPONDENCE_GEOMETRY_REVIEW_20261117.json':Path('/workspace/hud04-multipatch-review-20261117/CONTACT_CORRESPONDENCE_GEOMETRY_REVIEW_20261117.json'),
    }
    for name,path in evidence.items():files['evidence/'+name]=path.read_bytes()
    readme='''# 原始动作与无损整理归档 — SOURCE ONLY / NOT TRAINING READY

这不是 HUD04 重定向参考动作，不可导入训练。
保存四段原始 G1 动作（共3428帧、90Hz）、配套原始 STL/YAML、无损按名称排列的派生数据、溯源与审查证据。
original/ 7个文件逐字节保持，source_staging/ 可逆但不是标准参考动作字段。
没有关节角裁剪、重采样、符号变换、根坐标变换或四元数重排。

Step1 997帧仍有261帧超出HUD04位置限值、165个超速区间，尚无验收通过的HUD04动作。
官方发布消费接口有torso/wxyz绑定；原始exporter执行、原始世界单位/对齐未完整恢复。
原metadata.yaml引用缺失的Step6；不得把它当作完整可用的数据集元数据。
metadata_kneelClimbStep1.yaml与配对Step1/STL链接已恢复。
历史报告保留原时点判断，不能把不同报告时间的进度混为最终验收。

accepted_reference=false; training_ready=false; retarget_solver_calls=0; ppo_updates=0。
头部yaw/pitch固定零，不进入动作；29身体动作与模型限值保持不变。
后续：完成目标几何/接触策略与有界重定向，再独立检查全序列及真实匹配地形回放。
'''
    files['README_zh.md']=readme.encode()
    assert sum(row['frames'] for row in counts)==3428
    archive_manifest=dict(schema_version=1,status='SOURCE_ONLY_NOT_HUD04_REFERENCE',
        original_file_count=7,source_motion_count=4,total_frames=3428,fps=90,motions=counts,
        body_actions=29,head_actions=0,original_bytes_preserved=True,lossless_roundtrip_verified=True,
        accepted_reference=False,training_ready=False,retarget_solver_calls=0,ppo_updates=0,
        files=[dict(path=name,bytes=len(raw),sha256=digest(raw)) for name,raw in sorted(files.items())])
    encoded=json.dumps(archive_manifest,ensure_ascii=False,indent=2,allow_nan=False).encode()
    files['ARCHIVE_MANIFEST.json']=encoded
    archive=ROOT/'hud04-source-only-20261117.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,raw in sorted(files.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,11,17,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;z.writestr(info,raw)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(files)
        for name,raw in files.items():assert z.read(name)==raw
        assert json.loads(z.read('ARCHIVE_MANIFEST.json'))==archive_manifest
    for row in manifest['original_input_hashes_verified']:
        assert digest((RAW/row['name']).read_bytes())==row['sha256']
    for name,raw in [('ARCHIVE_MANIFEST.json',encoded),('README_zh.md',readme.encode())]:
        with (ROOT/name).open('xb') as f:f.write(raw)
    receipt=dict(archive=archive.name,bytes=archive.stat().st_size,sha256=digest(archive.read_bytes()),
        archive_entries=len(files),zip_readback_verified=True,original_hashes_rechecked=True,
        source_backup_complete=True,hud04_reference_complete=False,accepted_reference=False,training_ready=False)
    with (ROOT/'LOCAL_ARCHIVE_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()