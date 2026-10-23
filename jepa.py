"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        action_free_predictor=None,
        diff_encoder=None,
        diff_predictor=None,
        diff_transition_predictor=None,
        diff_latent_predictor=None,
        diff_action_embedding_predictor=None,
        image_decoder=None,
        pose_probe=None,
        pose_probe_linear=None,
        pose_probe_mlp=None,
        action_embedding_predictor=None,
        action_raw_predictor=None,
        causality_classifier=None,
        order_transition_module=None,
        projector=None,
        diff_projector=None,
        pred_proj=None,
        action_free_pred_proj=None,
        action_residual_pred_proj=None,
        diff_pred_proj=None,
        latent_normalize=None,
        latent_target_norm=None,
        predictor_outputs_delta=False,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.action_free_predictor = action_free_predictor
        self.diff_encoder = diff_encoder
        self.diff_predictor = diff_predictor
        self.diff_transition_predictor = diff_transition_predictor
        self.diff_latent_predictor = diff_latent_predictor
        self.diff_action_embedding_predictor = diff_action_embedding_predictor
        self.image_decoder = image_decoder
        self.pose_probe = pose_probe
        self.pose_probe_linear = pose_probe_linear
        self.pose_probe_mlp = pose_probe_mlp
        self.action_embedding_predictor = action_embedding_predictor
        self.action_raw_predictor = action_raw_predictor
        self.causality_classifier = causality_classifier
        self.order_transition_module = order_transition_module
        self.projector = projector or nn.Identity()
        self.diff_projector = diff_projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.action_free_pred_proj = action_free_pred_proj or nn.Identity()
        self.action_residual_pred_proj = action_residual_pred_proj or nn.Identity()
        self.diff_pred_proj = diff_pred_proj or nn.Identity()
        self.latent_normalize = latent_normalize
        self.latent_target_norm = latent_target_norm
        self.predictor_outputs_delta = predictor_outputs_delta

    def _normalize_latent(self, emb):
        if self.latent_normalize in (None, "none", False):
            return emb
        if self.latent_normalize in ("l2", "normalize"):
            return F.normalize(emb, dim=-1) * (emb.size(-1)**0.5)
        raise ValueError(f"Unknown latent_normalize option: {self.latent_normalize}")

    def _apply_target_latent_norm(self, emb):
        if self.latent_target_norm in (None, False):
            return emb
        if self.latent_target_norm <= 0:
            raise ValueError("latent_target_norm must be positive")
        return F.normalize(emb, dim=-1) * self.latent_target_norm

    def _encode_pixels(
        self,
        pixels,
        encoder=None,
        projector=None,
        output_attentions=False,
        apply_projector=True,
    ):
        encoder = encoder or self.encoder
        projector = projector or self.projector
        kwargs = {
            "interpolate_pos_encoding": True,
            "output_attentions": bool(output_attentions),
        }
        config = getattr(encoder, "config", None)
        previous_attn_implementation = getattr(config, "_attn_implementation", None)
        if (
            output_attentions
            and config is not None
            and hasattr(config, "_attn_implementation")
        ):
            config._attn_implementation = "eager"
        try:
            try:
                output = encoder(pixels, **kwargs)
            except TypeError:
                kwargs.pop("output_attentions", None)
                output = encoder(pixels, **kwargs)
        finally:
            if config is not None and previous_attn_implementation is not None:
                config._attn_implementation = previous_attn_implementation

        pixels_emb = output.last_hidden_state[:, 0]
        if apply_projector:
            emb = projector(pixels_emb)
            emb = self._normalize_latent(emb)
        else:
            emb = pixels_emb
        return emb, output

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        capture_attention = bool(info.get("return_attention", False))
        emb, output = self._encode_pixels(
            pixels,
            encoder=self.encoder,
            projector=self.projector,
            output_attentions=False,
        )
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        info["emb"] = self._apply_target_latent_norm(info["emb"])
        if capture_attention:
            with torch.no_grad():
                _, attention_output = self._encode_pixels(
                    pixels[:1],
                    encoder=self.encoder,
                    projector=self.projector,
                    output_attentions=True,
                    apply_projector=False,
                )
            if getattr(attention_output, "attentions", None) is not None:
                info["attentions"] = tuple(
                    attention.detach() for attention in attention_output.attentions
                )

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def encode_diff(self, diff_pixels):
        if self.diff_encoder is None:
            raise RuntimeError("diff_encoder is not configured")
        b = diff_pixels.size(0)
        diff_pixels = rearrange(diff_pixels.float(), "b t ... -> (b t) ...")
        diff_emb, _ = self._encode_pixels(
            diff_pixels,
            encoder=self.diff_encoder,
            projector=self.diff_projector,
            output_attentions=False,
        )
        return rearrange(diff_emb, "(b t) d -> b t d", b=b)

    def reconstruct(self, emb):
        if self.image_decoder is None:
            raise RuntimeError("image_decoder is not configured")
        return self.image_decoder(emb.detach())

    def probe_pose(self, emb):
        probes = {}
        if self.pose_probe_linear is not None:
            probes["linear"] = self.pose_probe_linear(emb.detach())
        if self.pose_probe_mlp is not None:
            probes["mlp"] = self.pose_probe_mlp(emb.detach())
        if probes:
            return probes
        if self.pose_probe is None:
            raise RuntimeError("pose_probe is not configured")
        return {"mlp": self.pose_probe(emb.detach())}

    def predict_action_emb(self, z_t, z_t1):
        if self.action_embedding_predictor is None:
            raise RuntimeError("action_embedding_predictor is not configured")
        return self.action_embedding_predictor(z_t, z_t1)

    def predict_action_raw(self, delta_z):
        if self.action_raw_predictor is None:
            raise RuntimeError("action_raw_predictor is not configured")
        return self.action_raw_predictor(delta_z)

    def classify_causality(self, delta_z, action):
        if self.causality_classifier is None:
            raise RuntimeError("causality_classifier is not configured")
        return self.causality_classifier(delta_z, action)

    def order_transition_loss(self, emb, act_emb):
        if self.order_transition_module is None:
            raise RuntimeError("order_transition_module is not configured")
        return self.order_transition_module(emb, act_emb)

    def predict_delta(self, emb, act_emb):
        """Return the raw predictor output before optional delta-to-state conversion.
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        if self.action_free_predictor is not None:
            baseline, residual = self.predict_action_residual(emb, act_emb)
            return baseline + residual
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def predict_action_residual(self, emb, act_emb):
        """Return B(z_history) and P(z_history, action_history)."""
        if self.action_free_predictor is None:
            raise RuntimeError("action_free_predictor is not configured")
        baseline = self.action_free_predictor(emb)
        baseline = self.action_free_pred_proj(
            rearrange(baseline, "b t d -> (b t) d")
        )
        baseline = rearrange(baseline, "(b t) d -> b t d", b=emb.size(0))
        residual = self.predict_action_component(emb, act_emb)
        return baseline, residual

    def predict_action_component(self, emb, act_emb):
        """Return only P(z_history, action_history)."""
        residual = self.predictor(emb, act_emb)
        residual = self.action_residual_pred_proj(
            rearrange(residual, "b t d -> (b t) d")
        )
        return rearrange(residual, "(b t) d -> b t d", b=emb.size(0))

    def predict(self, emb, act_emb):
        """Predict next state embedding.

        If predictor_outputs_delta is enabled, the raw predictor output is a
        latent delta and is converted to next-state latent here. This keeps
        rollout/planning callers on the usual next-state API.
        """
        if self.action_free_predictor is not None:
            baseline, residual = self.predict_action_residual(emb, act_emb)
            preds = baseline + residual
        else:
            preds = self.predict_delta(emb, act_emb)
        if self.predictor_outputs_delta:
            preds = emb + preds
        return preds

    def predict_diff(self, emb, act_emb):
        if self.diff_predictor is None:
            raise RuntimeError("diff_predictor is not configured")
        preds = self.diff_predictor(emb, act_emb)
        preds = self.diff_pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def predict_diff_transition(self, z_t, d_t):
        if self.diff_transition_predictor is None:
            raise RuntimeError("diff_transition_predictor is not configured")
        return self.diff_transition_predictor(z_t, d_t)

    def predict_diff_from_latents(self, z_t, z_t1):
        if self.diff_latent_predictor is None:
            raise RuntimeError("diff_latent_predictor is not configured")
        return self.diff_latent_predictor(z_t, z_t1)

    def predict_action_emb_from_diff(self, d_t):
        if self.diff_action_embedding_predictor is None:
            raise RuntimeError("diff_action_embedding_predictor is not configured")
        return self.diff_action_embedding_predictor(d_t)

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_emb = self.action_encoder(act)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
