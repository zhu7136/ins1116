"""Exact, default-on train.py gate patch; planning never writes or starts Isaac.

Only pristine audited bytes or our exact already-patched bytes are accepted.
Existing different helpers/receipt and symlink destinations are never overwritten.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN = "scripts/instinct_rl/train.py"
TRAIN_SHA256 = "8a38599e47ce94940d6488c89981c4ac1dd601e400ea5cb96e8b67d1ffdeaeb2"
RECEIPT = "scripts/instinct_rl/hud04_startup_install.json"
HELPERS = {"startup_gate.py": "scripts/instinct_rl/hud04_startup_gate.py",
           "runtime_gate.py": "scripts/instinct_rl/hud04_runtime_gate.py"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Audited train.py patch anchor changed")
    return source.replace(old, new, 1)


def patch_train(raw):
    text = raw.decode()
    text = _replace(text, "from isaaclab.app import AppLauncher",
        "# HUD04_STARTUP_GATE_V1: required before any Isaac/AppLauncher import.\n"
        "from hud04_startup_gate import prelaunch as _hud04_prelaunch\n"
        "_hud04_gate = _hud04_prelaunch(sys.argv[1:], __file__)\n\n"
        "from isaaclab.app import AppLauncher")
    text = _replace(text,
        'parser = argparse.ArgumentParser(description="Train an RL agent with Instinct-RL.")',
        'parser = argparse.ArgumentParser(description="Train an RL agent with Instinct-RL.", allow_abbrev=False)')
    text = _replace(text, "    # create runner from instinct-rl\n",
        "    # Fail closed after a normal full-task reset, before runner/checkpoint/learn.\n"
        "    if _hud04_gate is not None:\n"
        "        _hud04_gate.validate_environment(env)\n"
        "        _hud04_gate.require_runner_permission()\n\n"
        "    # create runner from instinct-rl\n")
    text = _replace(text, "    # close the simulator\n    env.close()",
        "    # close the simulator\n"
        "    if _hud04_gate is None:\n"
        "        env.close()\n"
        "    else:\n"
        "        _hud04_gate.close_environment()")
    text = _replace(text,
        '    # run the main function\n    main()\n    # close sim app\n    simulation_app.close()',
        '    if _hud04_gate is None:\n'
        '        main()\n'
        '        simulation_app.close()\n'
        '    else:\n'
        '        try:\n'
        '            main()\n'
        '        finally:\n'
        '            _hud04_gate.close_application(simulation_app)')
    compile(text, TRAIN, "exec")
    return text.encode()


def plan(upstream, overlay_files):
    """Return [(target, desired_bytes, already_identical)], fully checked first."""
    upstream = Path(upstream).resolve()
    target = upstream / TRAIN
    raw = target.read_bytes()
    if sha(raw) == TRAIN_SHA256:
        patched = patch_train(raw)
    else:
        # Recover audited original from git, then compare the COMPLETE patched file.
        import subprocess
        original = subprocess.check_output(["git", "-C", str(upstream), "show", "HEAD:" + TRAIN])
        if sha(original) != TRAIN_SHA256:
            raise ValueError("Audited original train.py identity mismatch")
        patched = patch_train(original)
        if raw != patched:
            raise ValueError("Refuse modified or partially patched train.py")
    payloads = {TRAIN: patched}
    for source, destination in HELPERS.items():
        raw_helper = (ROOT / source).read_bytes()
        compile(raw_helper, source, "exec")
        payloads[destination] = raw_helper
    for source, destination, _ in overlay_files:
        payloads[str(destination.relative_to(upstream))] = source.read_bytes()
    receipt = {"schema_version": 1, "upstream_commit": "de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057",
               "files": {k: sha(v) for k, v in sorted(payloads.items())}}
    # Overlay files are handled by the additive installer, but included in identity receipt.
    changes = {k: v for k, v in payloads.items() if k == TRAIN or k in HELPERS.values()}
    changes[RECEIPT] = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    result = []
    for relative, desired in changes.items():
        path = upstream / relative
        if not path.resolve().is_relative_to(upstream) or any(p.is_symlink() for p in [path, *path.parents]):
            raise ValueError("Symlink/escaping gate destination rejected")
        exists = path.exists()
        identical = exists and path.read_bytes() == desired
        if exists and not identical and relative != TRAIN:
            raise FileExistsError("Refuse existing different gate helper/receipt: " + relative)
        result.append((path, desired, identical))
    return result


def apply(changes):
    for path, desired, identical in changes:
        if not identical:
            path.parent.mkdir(parents=True, exist_ok=True)
            # train.py is the sole authorized source replacement; all helpers exclusive-create.
            if path.name == "train.py":
                if sha(path.read_bytes()) != TRAIN_SHA256:
                    raise ValueError("train.py changed after plan; stop")
                path.write_bytes(desired)
            else:
                with path.open("xb") as stream:
                    stream.write(desired)
        if path.read_bytes() != desired:
            raise RuntimeError("Startup gate installation readback mismatch")