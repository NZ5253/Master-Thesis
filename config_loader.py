# config_loader.py
import yaml


def load_env_config(scenario: str = "parallel"):
    """
    scenario: "parallel" or "perpendicular"
    returns a dict ready to pass to ParkingEnv(...)
    """
    with open("config_env.yaml", "r") as f:
        full_cfg = yaml.safe_load(f)

    base = {
        "vehicle": full_cfg["vehicle"],
        "dt": full_cfg.get("dt", 0.1),
        "max_steps": full_cfg.get("max_steps", 200),
        "world": full_cfg.get("world", {"width": 4.0, "height": 4.0}),
    }

    scen = full_cfg["scenarios"][scenario]
    base["goal"] = scen["goal"]
    base["obstacles"] = scen["obstacles"]

    return base


# from env.parking_env import ParkingEnv
# from config_loader import load_env_config
#
# cfg = load_env_config("parallel")  # or "perpendicular"
# env = ParkingEnv(cfg)
