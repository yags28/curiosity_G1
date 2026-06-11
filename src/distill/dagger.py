"""DAgger policy distillation: student imitates trained teacher actor."""

import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.networks import MLP

_SUCCESS_KEY: dict[str, str] = {
    "distant_target":     "tool_contact_events",
    "elevated_button":    "button_contact_events",
    "occluded_retrieval": "tool_contact_events",
    "weight_lever":       "lever_contact_events",
    "composite":          "button_contact_events",
}


class StudentPolicy(nn.Module):
    """Standalone actor for deployment — no critics, no curiosity.

    With tanh=True the action is squashed to (-1, 1): forward() returns
    tanh(logits) and raw() exposes the pre-squash logits for a saturation
    penalty. Param layout is identical either way, so checkpoints are
    cross-compatible (the tanh flag lives in the checkpoint dict).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
                 tanh: bool = False):
        super().__init__()
        self.mean_net = MLP(obs_dim, [hidden_dim, hidden_dim], action_dim, out_std=0.01)
        self.logstd   = nn.Parameter(torch.zeros(1, action_dim))
        self.tanh     = tanh

    def raw(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_net(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        a = self.mean_net(obs)
        return torch.tanh(a) if self.tanh else a


def _load_teacher(ckpt_path: str, obs_dim: int, action_dim: int,
                  hidden_dim: int, device: str) -> nn.Module:
    """Load only actor_mean weights from a PPO checkpoint; returns frozen MLP."""
    ckpt = torch.load(ckpt_path, map_location=device)
    # Extract actor_mean sub-keys, skipping critics (different input dim)
    actor_sd = {
        k[len("actor_mean."):]: v
        for k, v in ckpt["actor_critic"].items()
        if k.startswith("actor_mean.")
    }
    net = MLP(obs_dim, [hidden_dim, hidden_dim], action_dim, out_std=0.01)
    net.load_state_dict(actor_sd)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class DAggerDistiller:
    def __init__(self, cfg: dict, teacher_ckpt: str,
                 obs_dim: int, action_dim: int, device: str):
        self.cfg        = cfg
        self.device     = device
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        d = cfg["dagger"]
        self.num_iterations = d["num_iterations"]
        self.collect_steps  = d["collect_steps"]
        self.train_epochs   = d["train_epochs"]
        self.batch_size     = d["batch_size"]
        self.beta           = d["beta_start"]
        self.beta_min       = d["beta_min"]
        self.beta_decay     = d["beta_decay"]
        hidden_dim          = d.get("hidden_dim", 256)
        self.tanh           = d.get("tanh_squash", False)
        self.sat_penalty    = d.get("saturation_penalty", 0.0)
        # A tanh student can only match targets inside (-1,1), so always
        # imitate the clamped teacher action (what actually drives the robot).
        self.imitate_clamped = d.get("imitate_clamped", self.tanh)

        self.teacher = _load_teacher(teacher_ckpt, obs_dim, action_dim, hidden_dim, device)
        self.student  = StudentPolicy(obs_dim, action_dim, hidden_dim, tanh=self.tanh).to(device)
        self.opt      = optim.Adam(self.student.parameters(), lr=d["lr"])

        self._obs_buf: list[torch.Tensor] = []
        self._act_buf: list[torch.Tensor] = []

    def _collect(self, env, use_student: bool) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Roll out for collect_steps steps; return (obs, teacher_labels, success_rate)."""
        success_key = _SUCCESS_KEY.get(self.cfg["task"], "tool_contact_events")
        all_obs: list[torch.Tensor] = []
        all_act: list[torch.Tensor] = []
        ep_succ: list[float]         = []
        run_suc = torch.zeros(env.num_envs, device=self.device)

        obs_dict, _ = env.reset()
        pol_obs = obs_dict["policy"]

        for _ in range(self.collect_steps):
            with torch.no_grad():
                teacher_mean = self.teacher(pol_obs)
                step_action  = self.student(pol_obs) if use_student else teacher_mean

            all_obs.append(pol_obs.clone())
            all_act.append(teacher_mean.clone())

            obs_dict, _, terminated, timed_out, extras = env.step(step_action.clamp(-1.0, 1.0))
            pol_obs = obs_dict["policy"]
            done    = (terminated | timed_out).float()
            run_suc = torch.clamp(run_suc + extras.get(success_key, 0.0), 0.0, 1.0)
            for i in (done > 0.5).nonzero(as_tuple=True)[0]:
                ep_succ.append(float(run_suc[i].item() > 0))
                run_suc[i] = 0.0

        obs_t = torch.cat(all_obs, dim=0)   # [collect_steps * num_envs, obs_dim]
        act_t = torch.cat(all_act, dim=0)
        succ  = float(np.mean(ep_succ)) if ep_succ else 0.0
        return obs_t, act_t, succ

    def _train(self) -> tuple[float, float, float]:
        """Supervised update on aggregated dataset.

        Returns (imitation_loss, saturation_loss, mean_abs_action).
        """
        obs = torch.cat(self._obs_buf, dim=0)
        act = torch.cat(self._act_buf, dim=0)
        N   = obs.shape[0]
        imit_losses, sat_losses, abs_acts = [], [], []

        for _ in range(self.train_epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, self.batch_size):
                mb     = perm[start : start + self.batch_size]
                logits = self.student.raw(obs[mb])
                pred   = torch.tanh(logits) if self.tanh else logits
                target = act[mb].clamp(-1.0, 1.0) if self.imitate_clamped else act[mb]

                imit = ((pred - target) ** 2).mean()
                sat  = self.sat_penalty * (logits ** 2).mean()
                loss = imit + sat

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                imit_losses.append(imit.item())
                sat_losses.append(sat.item())
                abs_acts.append(pred.abs().mean().item())

        return (float(np.mean(imit_losses)), float(np.mean(sat_losses)),
                float(np.mean(abs_acts)))

    def run(self, env) -> None:
        task     = self.cfg["task"]
        seed     = self.cfg["seed"]
        tag      = f"_tanh_p{int(round(self.sat_penalty * 1000)):03d}" if self.tanh else ""
        run_name = f"dagger{tag}__{task}__seed{seed}"
        csv_dir  = os.path.join("logs", run_name)
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "metrics.csv")
        ckpt_dir = os.path.join("checkpoints", run_name)
        os.makedirs(ckpt_dir, exist_ok=True)

        fields = ["iteration", "beta", "samples", "imit_loss", "sat_loss",
                  "mean_abs_action", "success_rate"]
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

        print(f"[dagger] run={run_name} | iters={self.num_iterations} | "
              f"tanh={self.tanh} | sat_penalty={self.sat_penalty} | "
              f"collect_steps={self.collect_steps} | envs={env.num_envs}")

        for it in range(self.num_iterations):
            obs, act, succ = self._collect(env, use_student=(it > 0))
            self._obs_buf.append(obs)
            self._act_buf.append(act)
            total = sum(b.shape[0] for b in self._obs_buf)

            imit, sat, mabs = self._train()

            print(
                f"[dagger] iter={it+1:3d}/{self.num_iterations} | "
                f"β={self.beta:.2f} | samples={total:,} | imit={imit:.4f} | "
                f"sat={sat:.4f} | |a|={mabs:.3f} | success={succ:.2%}"
            )

            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow({
                    "iteration":       it + 1,
                    "beta":            self.beta,
                    "samples":         total,
                    "imit_loss":       imit,
                    "sat_loss":        sat,
                    "mean_abs_action": mabs,
                    "success_rate":    succ,
                })

            torch.save(
                {"iteration": it + 1, "student": self.student.state_dict(),
                 "tanh": self.tanh},
                os.path.join(ckpt_dir, f"iter_{it+1:03d}.pt"),
            )

            self.beta = max(self.beta_min, self.beta * self.beta_decay)

        print(f"[dagger] complete — csv: {csv_path}")
