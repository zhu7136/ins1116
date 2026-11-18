"""Mandatory HUD04 train.py startup gate. Stdlib only until CPU preflight.

No AppLauncher, solver, runner or GPU submission here. Installed beside train.py.
Default check-only mode exits successfully ONLY after live-interface validation;
to allow runner construction later, explicitly set HUD04_STARTUP_MODE=train.
Neither mode manufactures motion acceptance or claims training/physics readiness.
"""
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys

TASKS = {"Instinct-Perceptive-Shadowing-HUD04-Plane-v0": "plane",
         "Instinct-Perceptive-Shadowing-HUD04-Terrain-v0": "terrain"}
PREFIX = "source/instinctlab/instinctlab/"


def require(condition, message):
    if not condition:
        raise ValueError("HUD04 startup gate: " + message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_source(name):
    """Find source WITHOUT importing package __init__ or initializing Isaac."""
    search = None
    parts = name.split(".")
    for i in range(len(parts)):
        fullname = ".".join(parts[:i + 1])
        spec = None
        for finder in sys.meta_path:
            method = getattr(finder, "find_spec", None)
            if method:
                spec = method(fullname, search)
                if spec is not None:
                    break
        require(spec is not None, "Cannot resolve source before AppLauncher: " + fullname)
        if i < len(parts) - 1:
            search = spec.submodule_search_locations
            require(search is not None, "Not a package: " + fullname)
    path = Path(spec.origin or "")
    require(path.suffix == ".py" and path.is_file(), "Inspectable Python source required: " + name)
    return path.resolve()


def check_source_paths(pins):
    paths = {name: resolve_source(name) for name in pins}
    for name, path in paths.items():
        require(digest(path) == pins[name], "Selected source mismatch before AppLauncher: " + name)
    return paths


def installed_receipt(train_path):
    train_path = Path(train_path).resolve()
    upstream = train_path.parents[2]
    receipt_path = train_path.with_name("hud04_startup_install.json")
    receipt = json.loads(receipt_path.read_text())
    required = {"scripts/instinct_rl/train.py", "scripts/instinct_rl/hud04_startup_gate.py",
                "scripts/instinct_rl/hud04_runtime_gate.py",
                PREFIX + "assets/hud04_contract.py", PREFIX + "assets/hud04_motion.py"}
    require(receipt.get("schema_version") == 1 and required.issubset(receipt.get("files", {})),
            "Incomplete installed gate identity receipt")
    for relative, expected in receipt["files"].items():
        path = upstream / relative
        require(not Path(relative).is_absolute() and path.resolve().is_relative_to(upstream)
                and digest(path) == expected, "Installed source changed: " + relative)
    return upstream, receipt_path


def input_receipt(upstream, kind):
    """Reuse actual CPU model/accepted-manifest/strict-997-frame/terrain gates."""
    assets = upstream / PREFIX / "assets"
    contract = load_file("_hud04_startup_contract", assets / "hud04_contract.py")
    motion = load_file("_hud04_startup_motion", assets / "hud04_motion.py")
    require(os.environ.get("HUMANOID_DESCRIPTION_DIR"), "HUMANOID_DESCRIPTION_DIR required")
    require(os.environ.get("HUD04_REFERENCE_MANIFEST"), "HUD04_REFERENCE_MANIFEST required; no fallback")
    urdf = (Path(os.environ["HUMANOID_DESCRIPTION_DIR"]).expanduser() /
            "HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf").resolve()
    manifest = Path(os.environ["HUD04_REFERENCE_MANIFEST"]).expanduser().resolve()
    names, limits = contract.validate_model(urdf)
    root, _ = contract.validate_reference_manifest(manifest, urdf,
        accepted_status="accepted_for_" + kind + "_replay")
    document = json.loads(manifest.read_text())
    paths = [manifest, urdf]
    for record in document["files"]:
        path = (root / record["path"]).resolve()
        motion.read_named_motion(path, names, limits)
        paths.append(path)
    if kind == "terrain":
        motion.validate_terrain_bundle(manifest)
        paths += [(root / document[key]["path"]).resolve()
                  for key in ("terrain_metadata", "terrain_mesh", "support_schedule")]
    for module_kind in ("asset", "camera", "control"):
        # Hash only: asset_config imports Isaac and MUST NOT execute pre-AppLauncher.
        # The two CPU-only helpers are loaded separately below/inside validate_model.
        relative, expected = contract.VERIFIED_MODULES[module_kind]
        baseline = Path(os.environ["HUD04_DELIVERY_ROOT"]).expanduser().resolve() / relative
        require(digest(baseline) == expected, "Baseline source changed: " + relative)
        paths.append(baseline)
    camera = contract.load_verified_module("camera")
    camera.chest_camera_mount(str(urdf), os.environ)
    control = contract.load_verified_module("control")
    return {str(path): digest(path) for path in paths}, names, control.control_values, root


class StartupGate:
    def __init__(self, train_path, kind, mode):
        self.train_path, self.kind, self.mode = Path(train_path).resolve(), kind, mode
        self.upstream, self.receipt_path = installed_receipt(train_path)
        self.runtime = load_file("_hud04_startup_runtime", self.train_path.with_name("hud04_runtime_gate.py"))
        self.sources = check_source_paths(self.runtime.SOURCE_PINS)
        self.inputs, self.names, self.control_values, self.motion_root = input_receipt(self.upstream, kind)
        self.receipt_digest = digest(self.receipt_path)
        self.wrapper = None
        self.validated = False

    def validate_environment(self, wrapper):
        """Called unconditionally by patched HUD04 train.py, BEFORE OnPolicyRunner."""
        self.wrapper = wrapper
        try:
            require(not self.validated, "Gate cannot be reused")
            installed_receipt(self.train_path)
            require(digest(self.receipt_path) == self.receipt_digest, "Installation receipt changed during startup")
            inputs, _, _, root = input_receipt(self.upstream, self.kind)
            require(inputs == self.inputs, "Reference/model/baseline bytes changed during startup")
            base = wrapper.unwrapped
            expected_class = "HUD04Perceptive" + self.kind.title() + "EnvCfg"
            require(type(base.cfg).__name__ == expected_class, "Wrong full-task config/Play override")
            require(type(base.cfg).__module__.startswith(
                "instinctlab.tasks.shadowing.perceptive.config.hud04."), "Wrong task config implementation")
            buffers = base.cfg.scene.motion_reference.motion_buffers
            key = "AMASSMotion" if self.kind == "plane" else "TerrainMotion"
            require(list(buffers) == [key], "Reference buffer replaced or extra buffers configured")
            cfg = buffers[key]
            require(Path(cfg.path).resolve() == root, "Live reference path != accepted input")
            expected_loader = "HUD04AmassMotion" if self.kind == "plane" else "HUD04TerrainMotion"
            require(cfg.class_type.__name__ == expected_loader and cfg.class_type.__module__ ==
                "instinctlab.motion_reference.motion_files.hud04_motion", "Strict named loader replaced")
            require(cfg.motion_interpolate_func is None and cfg.skip_frames == 0
                and not cfg.clip_joint_ref_to_robot_limits and not cfg.ensure_link_below_zero_ground
                and cfg.motion_start_height_offset == 0, "Reference transformations/clipping enabled")
            # Wrapper construction already resets once but discards that observation.
            # Use its real public reset, not compute()/get_observations() which pushes history.
            _, extras = wrapper.reset()
            report = self.runtime.check_live_environment(base, wrapper, extras["observations"],
                                                          self.names, self.control_values)
            require(report.get("runtime_interface_checks_passed") is True, "No explicit runtime pass")
            for name, expected in self.sources.items():
                selected = report["selected_loaded_sources"][name]
                require(Path(selected["path"]).resolve() == expected,
                        "Loaded module != prelaunch resolved source: " + name)
            self.validated = True
            report.update(startup_gate_passed=True, startup_mode=self.mode,
                          runner_allowed=self.mode == "train", physics_acceptance_proven=False,
                          training_ready=False)
            print("HUD04_STARTUP_GATE=" + json.dumps(report, sort_keys=True), flush=True)
            if self.mode == "check-only":
                raise SystemExit(0)
            return report
        except BaseException as failure:
            # No conversion to warning/fallback; original failure propagates.
            try:
                self.close_environment()
            except Exception:
                if isinstance(failure, SystemExit) and failure.code == 0:
                    raise RuntimeError("HUD04 check-only passed but environment cleanup failed") from failure
                print("HUD04 environment cleanup also failed", file=sys.stderr, flush=True)
            raise

    def require_runner_permission(self):
        require(self.validated is True and self.mode == "train", "Runner construction not permitted")

    def close_environment(self):
        if self.wrapper is not None:
            wrapper, self.wrapper = self.wrapper, None
            try:
                wrapper.close()
            except Exception as exc:
                print("HUD04 cleanup failed: " + type(exc).__name__, file=sys.stderr, flush=True)
                raise

    def close_application(self, app):
        # Public shutdown sequence previously verified for this project.
        # May also be called when env construction failed before assigning wrapper.
        try:
            try:
                self.close_environment()
            finally:
                from isaaclab.sim import SimulationContext
                SimulationContext.clear_instance()
        finally:
            app.close(wait_for_replicator=False)


def prelaunch(argv, train_path):
    """Inspect ORIGINAL CLI, before Hydra removes it; no bypass/skip option."""
    tasks, num_envs = [], []
    for i, arg in enumerate(argv):
        for flag, values in (("--task", tasks), ("--num_envs", num_envs)):
            if arg == flag:
                require(i + 1 < len(argv), "Missing " + flag)
                values.append(argv[i + 1])
            elif arg.startswith(flag + "="):
                values.append(arg.split("=", 1)[1])
    if not any("hud04" in str(arg).lower() for arg in argv):
        return None  # Original G1/non-HUD04 behavior, except disabled CLI abbreviation.
    require(len(tasks) == 1 and tasks[0] in TASKS, "Exactly one supported HUD04 Plane/Terrain train task required")
    require(len(num_envs) <= 1 and all(value == "2" for value in num_envs), "Initial full-task gate requires two envs")
    require(not any(arg.split("=", 1)[0] in ("--distributed", "--video", "--debug") for arg in argv)
            and "LOCAL_RANK" not in os.environ, "Distributed/video/debug startup not audited")
    require(not any("=" in arg and not arg.startswith("--") for arg in argv),
            "Hydra overrides are disabled for initial HUD04 gate; review config changes explicitly")
    mode = os.environ.get("HUD04_STARTUP_MODE", "check-only")
    require(mode in ("check-only", "train"), "HUD04_STARTUP_MODE must be check-only or train")
    return StartupGate(train_path, TASKS[tasks[0]], mode)