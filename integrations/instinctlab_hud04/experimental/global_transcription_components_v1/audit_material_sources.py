"""One exclusive static evidence output; NO material getter or simulator call."""
import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup

from material_evidence import audit_source


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    upstream = Path('/workspace/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py')
    cfg = repo/'integrations/instinctlab_hud04/overlay/instinctlab/tasks/shadowing/perceptive/config/hud04'
    output = here/'MATERIAL_SOURCE_EVIDENCE.json'
    if output.exists():
        raise FileExistsError('Evidence already exists; no replay/overwrite')
    report = audit_source(upstream, cfg/'perceptive_shadowing_cfg.py',
                          cfg/'terrain_shadowing_cfg.py', repo/'flux/hud04_asset_camera_probe/asset_config.py')
    docs = Path('/workspace/hud04-nvidia1073-docs-20261118')
    receipt = json.loads((docs/'tensors.receipt.json').read_text())
    raw = (docs/'tensors.html').read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt['sha256']:
        raise ValueError('Pinned official documentation bytes changed')
    soup = BeautifulSoup(raw, 'html.parser')
    anchor = 'omni.physics.tensors.impl.api.ArticulationView.get_material_properties'
    entry = soup.find(id=anchor)
    if entry is None:
        raise ValueError('Official material getter entry missing')
    text = entry.parent.get_text(' ', strip=True)
    for phrase in ('(count, max_shapes, 3)', 'static friction', 'dynamic friction', 'restitution'):
        if phrase not in text:
            raise ValueError('Material getter schema not found in official entry')
    report['passive_getter_documentation'] = {
        'url': receipt['url'] + '#' + anchor,
        'local_html_sha256': receipt['sha256'],
        'shape': ['count', 'max_shapes', 3],
        'columns': ['static_friction', 'dynamic_friction', 'restitution'],
        'does_not_supply': ['slot-to-collider mapping', 'padding mask', 'terrain material',
                            'combine modes and pair priority', 'effective pair friction'],
        'actual_runtime_getter_calls_this_stage': 0,
    }
    report['audit_implementation_sha256'] = {
        name: hashlib.sha256((here/name).read_bytes()).hexdigest()
        for name in ('material_evidence.py', 'audit_material_sources.py', 'test_material_evidence.py')}
    report['gpu_submissions'] = 0
    report['optimizer_calls'] = 0
    report['new_motion_files'] = 0
    with output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()