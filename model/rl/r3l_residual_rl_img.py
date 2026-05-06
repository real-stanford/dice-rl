"""R3L-aligned residual head for DICE-RL image Robomimic post-training.

The rest of the training stack deliberately stays on the DICE-RL path
(RLPD, replay, n-step returns, TD target, optimizer scheduling).  This module
only changes the residual policy surface:
  - frozen base policy plus smooth bounded residual correction
  - optional final action clipping to the normalized action range
  - delayed best-of-N q-chunk action selection using the current critic
"""

from typing import Tuple

import torch

from model.rl.distill_residual_rl_img import DistillResidualRLImgModel


class R3LResidualRLImgModel(DistillResidualRLImgModel):
    def __init__(
        self,
        *args,
        max_correction: float = 0.15,
        clip_final_action: bool = True,
        action_clip_min: float = -1.0,
        action_clip_max: float = 1.0,
        q_chunk_num_samples: int = 4,
        q_chunk_critic_reduction: str = "min",
        q_chunk_warmup_steps: int = 50000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_correction = max_correction
        self.clip_final_action = clip_final_action
        self.action_clip_min = action_clip_min
        self.action_clip_max = action_clip_max
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
            raw_residual_actions = self.actor(state, pretrained_actions)
        else:
            raw_residual_actions = self.actor(state, noise)

        residual_actions = torch.tanh(raw_residual_actions) * self.max_correction
        total_actions = pretrained_actions + residual_actions
        if self.clip_final_action:
            total_actions = torch.clamp(
                total_actions,
                min=self.action_clip_min,
                max=self.action_clip_max,
            )

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
