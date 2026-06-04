from src.curiosity.rnd import RNDModule
from src.curiosity.drnd import DRNDModule
from src.curiosity.rdd import RDDModule


def make_curiosity(cfg: dict, obs_dim: int, device: str):
    """Factory: instantiate the curiosity module specified in cfg['curiosity']['method']."""
    method = cfg["curiosity"]["method"]
    c = cfg["curiosity"][method]
    out_dim    = c.get("output_dim", 64)
    hidden_dim = c.get("hidden_dim", 256)

    if method == "rnd":
        return RNDModule(obs_dim, output_dim=out_dim, hidden_dim=hidden_dim, device=device)
    elif method == "drnd":
        return DRNDModule(
            obs_dim,
            n_ensemble=c.get("n_ensemble", 5),
            output_dim=out_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
    elif method == "rdd":
        return RDDModule(
            obs_dim,
            n_ensemble=c.get("n_ensemble", 5),
            sigma=c.get("sigma", 1.0),
            output_dim=out_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
    else:
        raise ValueError(f"Unknown curiosity method: {method!r}. Choose rnd | drnd | rdd")
