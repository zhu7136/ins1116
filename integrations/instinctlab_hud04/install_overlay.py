"""Install HUD04 overlay AND mandatory train.py startup gate. Plan-only unless --apply.

Never clone, install dependencies, or launch a simulator. With --apply, the sole
source replacement is exact audited train.py -> reviewed mandatory gate patch.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_hud04_startup_patch", ROOT / "startup_patch.py")
STARTUP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(STARTUP)
PIN = "de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057"
AUDITED = {
    "tasks/shadowing/perceptive/perceptive_env_cfg.py": "ca9beeca2e9d59d1fbb588e35d2452261b171bdf9ebf85ba8efd786aa8e81504",
    "motion_reference/motion_reference_cfg.py": "494958c58aec758b33b9420feacaa9283652e528a3f8257232c185c4d094ac06",
    "motion_reference/motion_files/amass_motion_cfg.py": "a7b88cd0e783b65557a5a8bd4077813915504a9f736a16a08c59584159965fef",
}


def plan(upstream):
    upstream = Path(upstream).expanduser().resolve()
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != PIN:
        raise ValueError("InstinctLab HEAD must match audited commit " + PIN)
    package = upstream / "source/instinctlab/instinctlab"
    if not package.is_dir():
        raise ValueError("Expected an InstinctLab source checkout")
    for relative, expected in AUDITED.items():
        if hashlib.sha256((package / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Audited upstream source changed: " + relative)
    files = []
    for source in sorted((ROOT / "overlay/instinctlab").rglob("*.py")):
        relative = source.relative_to(ROOT / "overlay/instinctlab")
        target = package / relative
        # Existing symlink paths must not turn an additive install into a write elsewhere.
        if not target.resolve().is_relative_to(package.resolve()) or any(p.is_symlink() for p in [target, *target.parents]):
            raise ValueError("Symlink/escaping destination rejected")
        raw = source.read_bytes()
        compile(raw, str(source), "exec")
        if target.exists() and target.read_bytes() != raw:
            raise FileExistsError("Refuse to overwrite existing file: " + str(relative))
        files.append((source, target, target.exists()))
    expected = {
        "assets/hud04.py", "assets/hud04_contract.py", "assets/hud04_motion.py",
        "motion_reference/motion_files/hud04_motion.py",
        "tasks/shadowing/perceptive/config/hud04/__init__.py",
        "tasks/shadowing/perceptive/config/hud04/perceptive_shadowing_cfg.py",
        "tasks/shadowing/perceptive/config/hud04/terrain_shadowing_cfg.py",
        "tasks/shadowing/perceptive/config/hud04/conditional_contacts.py",
        "tasks/shadowing/perceptive/config/hud04/agents/__init__.py",
        "tasks/shadowing/perceptive/config/hud04/agents/instinct_rl_ppo_cfg.py",
    }
    if {str(t.relative_to(package)) for _, t, _ in files} != expected:
        raise ValueError("Unexpected overlay file set")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    files = plan(args.upstream)
    # Validate ALL destinations, identities and patch anchors before the first write.
    changes = STARTUP.plan(args.upstream, files)
    if args.apply:
        for source, target, exists in files:
            if not exists:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(source.read_bytes())
            if target.read_bytes() != source.read_bytes():
                raise RuntimeError("Installation readback mismatch")
        STARTUP.apply(changes)
        # Independent complete readback, including already-present identical files.
        if not all(identical for _, _, identical in STARTUP.plan(args.upstream, files)):
            raise RuntimeError("Gate install not idempotently verified")
    print(json.dumps(dict(mode="installed_and_verified" if args.apply else "plan_only",
        upstream_commit=PIN, additive_files=[str(t.relative_to(Path(args.upstream).expanduser().resolve())) for _, t, _ in files],
        startup_gate_files=[str(t.relative_to(Path(args.upstream).expanduser().resolve())) for t, _, _ in changes],
        startup_gate_default_on=True, startup_mode_default="check-only",
        gpu_started=False, training_updates=0,
        source_overwrites=sum(not identical and t.name == "train.py" for t, _, identical in changes)
            if args.apply else 0), indent=2))


if __name__ == "__main__":
    main()