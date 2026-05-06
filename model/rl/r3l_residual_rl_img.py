"""
R3L-aligned residual RL model for image Robomimic post-training.

This keeps the DICE-RL frozen image flow/diffusion policy interface, but changes
the online residual layer to match the QC-R3L training recipe:
  - frozen base policy plus bounded residual correction
  - query-chunk reward/discount semantics
  - conservative best-of-N q-chunk action selection after warmup
"""

from typing import Dict, Optional, Tuple

import torch

from model.rl.distill_residual_rl_img import DistillResidualRLImgModel


class R3LResidualRLImgModel(DistillResidualRLImgModel):
    def __init__(
        self,
        *args,
        max_correction: float = 0.15,
        r3l_query_discount_steps: int = 1,
        q_chunk_num_samples: int = 4,
        q_chunk_critic_reduction: str = "min",
        q_chunk_warmup_steps: int = 50000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_correction = max_correction
        self.r3l_query_discount_steps = r3l_query_discount_steps
        self.q_chunk_num_samples = q_chunk_num_samples
        self.q_chunk_critic_reduction = q_chunk_critic_reduction
        self.q_chunk_warmup_steps = q_chunk_warmup_steps

    def get_action(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
        return_pretrained_actions: bool = False,
    ):
        with torch.no_grad():
            output = self.pretrained_flow_policy.forward_from_features(
                features=state,
                init_noise=noise,
            )
            pretrained_actions = output.trajectories.detach()

        if self.condition_residual_on_base_action:
            residual_actions = self.actor(state, pretrained_actions)
        else:
            residual_actions = self.actor(state, noise)

        residual_actions = torch.clamp(
            residual_actions,
            min=-self.max_correction,
            max=self.max_correction,
        )
        total_actions = pretrained_actions + residual_actions

        if return_pretrained_actions:
            return total_actions, pretrained_actions
        return total_actions

    def get_exploration_action(
        self,
        state: torch.Tensor,
        num_samples: int = 4,
        exploration_strategy: str = "r3l_q_chunk",
        training_step: int = 0,
        replay_flow_model=None,
        replay_flow_config=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exploration_strategy != "r3l_q_chunk":
            return super().get_exploration_action(
                state=state,
                num_samples=num_samples,
                exploration_strategy=exploration_strategy,
                training_step=training_step,
                replay_flow_model=replay_flow_model,
                replay_flow_config=replay_flow_config,
            )

        batch_size = state.shape[0]
        device = state.device
        num_samples = max(1, int(num_samples or self.q_chunk_num_samples))

        if training_step <= self.q_chunk_warmup_steps or num_samples == 1:
            noise = torch.randn(batch_size, self.horizon_steps, self.action_dim, device=device)
            action = self.get_action(state, noise)
            return action, noise

        noise_samples = torch.randn(
            num_samples,
            batch_size,
            self.horizon_steps,
            self.action_dim,
            device=device,
        )
        state_flat = (
            state.unsqueeze(0)
            .expand(num_samples, -1, -1, -1)
            .reshape(num_samples * batch_size, *state.shape[1:])
        )
        noise_flat = noise_samples.reshape(
            num_samples * batch_size,
            self.horizon_steps,
            self.action_dim,
        )

        with torch.no_grad():
            actions_flat = self.get_action(state_flat, noise_flat)
            q_all = self.critic(state_flat, noise_flat, actions_flat, return_all=True)
            q_stacked = torch.stack(q_all, dim=0)
            q_stacked = q_stacked.view(len(q_all), num_samples, batch_size, 1)

            if self.q_chunk_critic_reduction == "min":
                q_scores = q_stacked.min(dim=0).values
            elif self.q_chunk_critic_reduction == "mean":
                q_scores = q_stacked.mean(dim=0)
            else:
                raise ValueError(
                    f"Unknown q_chunk_critic_reduction: {self.q_chunk_critic_reduction}"
                )

            selected = q_scores.squeeze(-1).argmax(dim=0)
            actions = actions_flat.view(
                num_samples,
                batch_size,
                self.horizon_steps,
                self.action_dim,
            )
            selected_actions = torch.stack(
                [actions[selected[b], b] for b in range(batch_size)]
            )
            selected_noise = torch.stack(
                [noise_samples[selected[b], b] for b in range(batch_size)]
            )

        return selected_actions, selected_noise

    def loss(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        gamma: float = 0.99,
        training_step: int = 0,
        q_overestimation: Optional[torch.Tensor] = None,
        n_steps: Optional[torch.Tensor] = None,
        data_source: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = state.shape[0]
        current_actions = self.get_action(state, noise)
        q_values = self.critic(state, noise, current_actions)

        with torch.no_grad():
            if not self.multi_sample_next_noise:
                next_noise = torch.randn(
                    batch_size,
                    action.shape[1],
                    self.action_dim,
                    device=self.device,
                )
                next_actions = self.get_action(next_state, next_noise)
                target_next_q = self.target_critic(next_state, next_noise, next_actions)
            else:
                k = self.num_next_noise_samples
                next_noise_samples = torch.randn(
                    k,
                    batch_size,
                    action.shape[1],
                    self.action_dim,
                    device=self.device,
                )
                next_state_rep = (
                    next_state.unsqueeze(0)
                    .expand(k, -1, -1, -1)
                    .reshape(k * batch_size, *next_state.shape[1:])
                )
                next_noise = next_noise_samples.reshape(
                    k * batch_size,
                    *next_noise_samples.shape[2:],
                )
                next_actions = self.get_action(next_state_rep, next_noise)
                target_q_samples = self.target_critic(
                    next_state_rep,
                    next_noise,
                    next_actions,
                )
                target_next_q = target_q_samples.reshape(k, batch_size, 1).mean(dim=0)

            if n_steps is None:
                query_steps = torch.ones_like(reward)
            else:
                query_steps = n_steps.float()
            gamma_effective = gamma ** (query_steps * self.r3l_query_discount_steps)
            target_q = reward + gamma_effective * (1 - done.float()) * target_next_q

        actor_losses = self.actor_loss(
            state,
            noise,
            current_actions,
            q_values,
            next_state,
            None,
            training_step,
            q_overestimation,
            data_source=data_source,
        )
        critic_losses = self.critic_loss(state, noise, action, target_q, data_source=data_source)

        total_loss = actor_losses["actor_total"] + self.critic_weight * critic_losses["critic_loss"]
        return {
            "total_loss": total_loss,
            "r3l/query_discount_mean": gamma_effective.mean(),
            "r3l/max_correction": torch.tensor(self.max_correction, device=self.device),
            **actor_losses,
            **critic_losses,
        }
