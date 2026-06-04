"""PPO + RND agent with asymmetric actor-critic for CDL tool-use tasks."""

import csv
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from src.utils.networks import ActorCritic
from src.curiosity import make_curiosity

# extras key each task uses to report per-step success fraction
_SUCCESS_KEY: dict[str, str] = {
    "distant_target":    "tool_contact_events",
    "elevated_button":   "button_contact_events",
    "occluded_retrieval":"tool_contact_events",
    "weight_lever":      "lever_contact_events",
    "composite":         "button_contact_events",
}


@dataclass
class RolloutBuffer:
    num_steps:      int
    num_envs:       int
    obs_dim:        int
    critic_obs_dim: int
    action_dim:     int
    device:         str

    obs_pol:  torch.Tensor = field(init=False)
    obs_crit: torch.Tensor = field(init=False)
    actions:  torch.Tensor = field(init=False)
    logprobs: torch.Tensor = field(init=False)
    rew_ext:  torch.Tensor = field(init=False)
    rew_int:  torch.Tensor = field(init=False)
    dones:    torch.Tensor = field(init=False)
    val_ext:  torch.Tensor = field(init=False)
    val_int:  torch.Tensor = field(init=False)

    def __post_init__(self):
        T, N, dev = self.num_steps, self.num_envs, self.device
        self.obs_pol  = torch.zeros(T, N, self.obs_dim,        device=dev)
        self.obs_crit = torch.zeros(T, N, self.critic_obs_dim, device=dev)
        self.actions  = torch.zeros(T, N, self.action_dim,     device=dev)
        self.logprobs = torch.zeros(T, N,                       device=dev)
        self.rew_ext  = torch.zeros(T, N,                       device=dev)
        self.rew_int  = torch.zeros(T, N,                       device=dev)
        self.dones    = torch.zeros(T, N,                       device=dev)
        self.val_ext  = torch.zeros(T, N,                       device=dev)
        self.val_int  = torch.zeros(T, N,                       device=dev)


def _compute_gae(
    rewards:    torch.Tensor,
    values:     torch.Tensor,
    dones:      torch.Tensor,
    next_value: torch.Tensor,
    next_done:  torch.Tensor,
    gamma:      float,
    lam:        float,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal = 1.0 - next_done.float()
            next_val    = next_value
        else:
            nonterminal = 1.0 - dones[t + 1]
            next_val    = values[t + 1]
        delta          = rewards[t] + gamma * next_val * nonterminal - values[t]
        gae            = delta + gamma * lam * nonterminal * gae
        advantages[t]  = gae
    return advantages, advantages + values


class PPOAgent:
    def __init__(self, cfg: dict, obs_dim: int, critic_obs_dim: int, action_dim: int, device: str):
        self.cfg    = cfg
        self.device = device

        p = cfg["ppo"]
        self.num_steps       = p["num_steps"]
        self.num_minibatches = p["num_minibatches"]
        self.update_epochs   = p["update_epochs"]
        self.lr              = p["lr"]
        self.anneal_lr       = p["anneal_lr"]
        self.gamma_int       = p["gamma_int"]
        self.gamma_ext       = p["gamma_ext"]
        self.gae_lambda      = p["gae_lambda"]
        self.clip_coef       = p["clip_coef"]
        self.ent_coef        = p["ent_coef"]
        self.vf_coef         = p["vf_coef"]
        self.max_grad_norm   = p["max_grad_norm"]

        method  = cfg["curiosity"]["method"]
        cur_cfg = cfg["curiosity"][method]
        rew_cfg = cfg["reward"]
        self.alpha              = rew_cfg["alpha_start"]
        self.alpha_start        = rew_cfg["alpha_start"]
        self.alpha_end          = rew_cfg["alpha_end"]
        self.alpha_anneal_steps = rew_cfg["alpha_anneal_steps"]

        hidden_dim    = cur_cfg.get("hidden_dim", 256)
        curiosity_lr  = cur_cfg.get("lr", 1e-4)

        self.ac        = ActorCritic(obs_dim, critic_obs_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.curiosity = make_curiosity(cfg, critic_obs_dim, device)

        self.opt_ppo = optim.Adam(self.ac.parameters(),                    lr=self.lr,      eps=1e-5)
        self.opt_rnd = optim.Adam(self.curiosity.predictor_parameters(),   lr=curiosity_lr, eps=1e-5)

        self.global_step = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _update_alpha(self):
        frac       = min(1.0, self.global_step / max(1, self.alpha_anneal_steps))
        self.alpha = self.alpha_start + frac * (self.alpha_end - self.alpha_start)

    def _update_lr(self, update: int, num_updates: int):
        frac = 1.0 - (update - 1) / num_updates
        for pg in self.opt_ppo.param_groups:
            pg["lr"] = frac * self.lr

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "global_step":    self.global_step,
            "actor_critic":   self.ac.state_dict(),
            "curiosity_state": self.curiosity.state_dict(),
            "opt_ppo":        self.opt_ppo.state_dict(),
            "opt_rnd":        self.opt_rnd.state_dict(),
            "alpha":          self.alpha,
        }, path)
        print(f"[ckpt] saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["actor_critic"])
        if "curiosity_state" in ckpt:
            self.curiosity.load_state_dict(ckpt["curiosity_state"])
        elif "rnd_predictor" in ckpt:
            # backward compat: old single-predictor RND checkpoints
            self.curiosity.predictor.load_state_dict(ckpt["rnd_predictor"])
        self.opt_ppo.load_state_dict(ckpt["opt_ppo"])
        self.opt_rnd.load_state_dict(ckpt["opt_rnd"])
        self.global_step = ckpt["global_step"]
        self.alpha       = ckpt["alpha"]
        print(f"[ckpt] resumed from {path} (step {self.global_step:,})")

    # ── main training loop ────────────────────────────────────────────────────

    def train(self, env, run=None):
        cfg          = self.cfg
        num_envs     = env.num_envs
        obs_dim      = env.cfg.observation_space
        critic_dim   = env.cfg.state_space
        action_dim   = env.cfg.action_space
        task         = cfg["task"]
        success_key  = _SUCCESS_KEY.get(task, "tool_contact_events")

        total_steps   = cfg["training"]["total_timesteps"]
        batch_size    = num_envs * self.num_steps
        minibatch_sz  = batch_size // self.num_minibatches
        num_updates   = total_steps // batch_size
        log_every     = cfg["training"]["log_interval"]
        ckpt_every    = cfg["training"]["checkpoint_interval"]
        ckpt_dir      = cfg["training"]["checkpoint_dir"]
        method        = cfg["curiosity"]["method"]
        run_name      = run.name if run else f"{task}__{method}__local"

        # ── CSV log setup (written regardless of wandb) ───────────────────────
        csv_dir  = os.path.join("logs", run_name)
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "metrics.csv")
        _CSV_FIELDS = [
            "global_step", "episode_reward", "intrinsic_reward", "success_rate",
            "episode_length", "alpha", "policy_loss", "value_loss", "entropy",
            "curiosity_loss", "approx_kl", "sps",
        ]
        with open(csv_path, "w", newline="") as _f:
            csv.DictWriter(_f, fieldnames=_CSV_FIELDS).writeheader()
        print(f"[launch] csv log   : {csv_path}")

        buf = RolloutBuffer(self.num_steps, num_envs, obs_dim, critic_dim, action_dim, self.device)

        obs_dict, _ = env.reset()
        next_pol    = obs_dict["policy"]
        next_crit   = obs_dict["critic"]
        next_done   = torch.zeros(num_envs, device=self.device)

        # Warm up curiosity obs running stats before first reward computation
        self.curiosity.obs_rms.update(next_crit)

        t0 = time.time()

        for update in range(1, num_updates + 1):
            if self.anneal_lr:
                self._update_lr(update, num_updates)

            # ── collect rollout ───────────────────────────────────────────────
            ep_rews: list[float] = []
            ep_lens: list[int]   = []
            ep_succ: list[float] = []
            run_rew = torch.zeros(num_envs, device=self.device)
            run_len = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
            run_suc = torch.zeros(num_envs, device=self.device)

            for step in range(self.num_steps):
                buf.obs_pol[step]  = next_pol
                buf.obs_crit[step] = next_crit
                buf.dones[step]    = next_done

                with torch.no_grad():
                    action, logprob, _, v_ext, v_int = self.ac.get_action_and_value(next_pol, next_crit)
                buf.actions[step]  = action
                buf.logprobs[step] = logprob
                buf.val_ext[step]  = v_ext
                buf.val_int[step]  = v_int

                obs_dict, rew_ext, terminated, timed_out, extras = env.step(action.clamp(-1.0, 1.0))

                # curiosity: update obs stats, compute intrinsic reward
                self.curiosity.obs_rms.update(obs_dict["critic"])
                rew_int = self.curiosity.compute_reward(obs_dict["critic"])

                buf.rew_ext[step] = rew_ext
                buf.rew_int[step] = rew_int

                done        = (terminated | timed_out).float()
                next_done   = done
                next_pol    = obs_dict["policy"]
                next_crit   = obs_dict["critic"]

                self.global_step += num_envs
                self._update_alpha()

                # per-env episode tracking
                run_rew += rew_ext
                run_len += 1
                run_suc  = torch.clamp(run_suc + extras.get(success_key, 0.0), 0.0, 1.0)
                for i in (done > 0.5).nonzero(as_tuple=True)[0]:
                    ep_rews.append(run_rew[i].item())
                    ep_lens.append(run_len[i].item())
                    ep_succ.append(float(run_suc[i].item() > 0))
                    run_rew[i] = 0.0
                    run_len[i] = 0
                    run_suc[i] = 0.0

            # ── compute advantages ────────────────────────────────────────────
            with torch.no_grad():
                nv_ext, nv_int = self.ac.get_value(next_crit)

            adv_ext, ret_ext = _compute_gae(
                buf.rew_ext, buf.val_ext, buf.dones, nv_ext, next_done, self.gamma_ext, self.gae_lambda)
            adv_int, ret_int = _compute_gae(
                buf.rew_int, buf.val_int, buf.dones, nv_int, next_done, self.gamma_int, self.gae_lambda)

            adv = adv_ext + self.alpha * adv_int
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # ── PPO + RND update ──────────────────────────────────────────────
            b_pol     = buf.obs_pol.reshape(-1, obs_dim)
            b_crit    = buf.obs_crit.reshape(-1, critic_dim)
            b_act     = buf.actions.reshape(-1, action_dim)
            b_logp    = buf.logprobs.reshape(-1)
            b_adv     = adv.reshape(-1)
            b_ret_ext = ret_ext.reshape(-1)
            b_ret_int = ret_int.reshape(-1)

            idx = torch.arange(batch_size, device=self.device)
            pg_losses, v_losses, ent_losses, curiosity_losses, kl_approx = [], [], [], [], []

            for _ in range(self.update_epochs):
                perm = idx[torch.randperm(batch_size, device=self.device)]
                for start in range(0, batch_size, minibatch_sz):
                    mb = perm[start : start + minibatch_sz]

                    _, new_lp, entropy, new_ve, new_vi = \
                        self.ac.get_action_and_value(b_pol[mb], b_crit[mb], b_act[mb])

                    ratio  = (new_lp - b_logp[mb]).exp()
                    mb_adv = b_adv[mb]

                    pg1     = -mb_adv * ratio
                    pg2     = -mb_adv * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()

                    v_loss  = 0.5 * ((new_ve - b_ret_ext[mb]) ** 2 +
                                     (new_vi - b_ret_int[mb]) ** 2).mean()
                    e_loss  = entropy.mean()
                    loss    = pg_loss + self.vf_coef * v_loss - self.ent_coef * e_loss

                    self.opt_ppo.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                    self.opt_ppo.step()

                    cur_loss = self.curiosity.predictor_loss(b_crit[mb])
                    self.opt_rnd.zero_grad()
                    cur_loss.backward()
                    self.opt_rnd.step()

                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    ent_losses.append(e_loss.item())
                    curiosity_losses.append(cur_loss.item())
                    kl_approx.append(((ratio - 1) - (new_lp - b_logp[mb])).mean().item())

            # ── logging ───────────────────────────────────────────────────────
            if update % log_every == 0:
                sps        = self.global_step / (time.time() - t0)
                mean_rew   = float(np.mean(ep_rews))   if ep_rews else 0.0
                mean_len   = float(np.mean(ep_lens))   if ep_lens else 0.0
                mean_succ  = float(np.mean(ep_succ))   if ep_succ else 0.0
                int_now    = buf.rew_int.mean().item()

                print(
                    f"[{update:5d}/{num_updates}] step={self.global_step:>10,} | "
                    f"rew={mean_rew:+.3f} | int={int_now:.4f} | "
                    f"succ={mean_succ:.2%} | α={self.alpha:.3f} | "
                    f"pg={np.mean(pg_losses):.4f} | sps={sps:.0f}"
                )

                _log_row = {
                    "global_step":    self.global_step,
                    "episode_reward": mean_rew,
                    "intrinsic_reward": int_now,
                    "success_rate":   mean_succ,
                    "episode_length": mean_len,
                    "alpha":          self.alpha,
                    "policy_loss":    float(np.mean(pg_losses)),
                    "value_loss":     float(np.mean(v_losses)),
                    "entropy":        float(np.mean(ent_losses)),
                    "curiosity_loss": float(np.mean(curiosity_losses)),
                    "approx_kl":      float(np.mean(kl_approx)),
                    "sps":            sps,
                }
                with open(csv_path, "a", newline="") as _f:
                    csv.DictWriter(_f, fieldnames=_CSV_FIELDS).writerow(_log_row)

                if run is not None:
                    wandb.log({
                        "global_step":        self.global_step,
                        "episode_reward":     mean_rew,
                        "intrinsic_reward":   int_now,
                        "extrinsic_reward":   mean_rew,
                        "success_rate":       mean_succ,
                        "episode_length":     mean_len,
                        "tool_contact_count": mean_succ * mean_len,
                        "alpha_annealing":    self.alpha,
                        "policy_loss":        float(np.mean(pg_losses)),
                        "value_loss":         float(np.mean(v_losses)),
                        "entropy":            float(np.mean(ent_losses)),
                        "approx_kl":          float(np.mean(kl_approx)),
                        "curiosity_loss":     float(np.mean(curiosity_losses)),
                        "reward_int_mean":    int_now,
                        "reward_int_std":     buf.rew_int.std().item(),
                        "sps":                sps,
                    }, step=self.global_step)

            # ── checkpoint ────────────────────────────────────────────────────
            if self.global_step % ckpt_every < batch_size:
                self.save(f"{ckpt_dir}/{run_name}/step_{self.global_step}.pt")

        print(f"[ppo] training complete — {self.global_step:,} steps")
