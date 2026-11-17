"""CPU preflight only. No simulator, dependency installation or Flux API calls."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("hud04_contract", ROOT / "overlay/instinctlab/assets/hud04_contract.py")
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--reference-manifest")
    args = parser.parse_args()
    report = dict(mode="plan_only", gpu_started=False, training_updates=0,
                  integration_gpu_validated=False, training_ready=False,
                  instinctlab_commit=contract.INSTINCTLAB_COMMIT)
    if args.check_model or args.reference_manifest:
        directory = os.environ.get("HUMANOID_DESCRIPTION_DIR")
        if not directory:
            raise RuntimeError("Set HUMANOID_DESCRIPTION_DIR to the prepared model checkout")
        urdf = Path(directory).expanduser() / "HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf"
        names, _ = contract.validate_model(urdf)
        report.update(mode="cpu_model_checked", body_actions=len(names), head_actions=0,
                      root_link=contract.ROOT_LINK, model_sha256=contract.sha256(urdf))
        if args.reference_manifest:
            _, records = contract.validate_reference_manifest(args.reference_manifest, urdf)
            report.update(mode="cpu_reference_checked", references=records,
                          physical_replay_independently_verified=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()