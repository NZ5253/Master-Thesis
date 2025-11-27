# config_loader.py
import yaml


def load_env_config(scenario: str = "perpendicular"):
    """
    scenario: "parallel" or "perpendicular"
    Returns a dict ready to pass to ParkingEnv(...).

    Supports two formats for config_env.yaml:

    1) New (with per-scenario block):
        vehicle: ...
        dt: ...
        max_steps: ...
        world: ...
        scenarios:
          perpendicular:
            goal: ...
            parking: ...
            spawn_lane: ...
            obstacles: ...

    2) Legacy / flat:
        vehicle: ...
        dt: ...
        max_steps: ...
        world: ...
        goal: ...
        parking: ...
        spawn_lane: ...
        obstacles: ...
    """
    with open("config_env.yaml", "r") as f:
        full_cfg = yaml.safe_load(f)

    # ---- CASE 1: flat / legacy config (no "scenarios") ----
    if "scenarios" not in full_cfg:
        # Assume file already is a complete env config.
        # Make sure a few top-level defaults exist, then return.
        cfg = dict(full_cfg)  # shallow copy

        if "world" not in cfg:
            cfg["world"] = {"width": 4.0, "height": 4.0}
        if "dt" not in cfg:
            cfg["dt"] = 0.1
        if "max_steps" not in cfg:
            cfg["max_steps"] = 200

        return cfg

    # ---- CASE 2: new config with "scenarios" ----
    base = {
        "vehicle": full_cfg["vehicle"],
        "dt": full_cfg.get("dt", 0.1),
        "max_steps": full_cfg.get("max_steps", 200),
        "world": full_cfg.get("world", {"width": 4.0, "height": 4.0}),
    }

    scen_dict = full_cfg["scenarios"]
    if scenario not in scen_dict:
        raise KeyError(f"Scenario '{scenario}' not found in config_env.yaml")

    scen = scen_dict[scenario]

    # Copy per-scenario blocks if present
    for key in ("goal", "parking", "spawn_lane", "obstacles"):
        if key in scen:
            base[key] = scen[key]

    return base
