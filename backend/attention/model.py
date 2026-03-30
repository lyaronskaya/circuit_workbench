import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


AVAILABLE_MODELS = {
    "gpt2-small": "gpt2-small",
    "pythia-2.8b": "pythia-2.8b",
}


class AttentionPatternExtractor:
    def __init__(self, model_name: str = "gpt2-small"):
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_name} not supported. Available models: {', '.join(AVAILABLE_MODELS.keys())}"
            )

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model {model_name} on {self.device}...")
        self.model = HookedTransformer.from_pretrained(
            AVAILABLE_MODELS[model_name],
            device=self.device,
            dtype=torch.float32,
        )
        self.model.eval()

        self.n_layers = self.model.cfg.n_layers
        self.n_heads = self.model.cfg.n_heads

        print(f"Model loaded: {model_name} with {self.n_layers} layers and {self.n_heads} heads")

    def get_attention_patterns(self, text: str) -> Dict[str, Any]:
        tokens_list = self.model.to_str_tokens(text)
        patterns: List[np.ndarray] = []

        def save_pattern(activation, hook):
            pattern = activation.detach().squeeze(0).cpu().numpy()
            patterns.append(pattern[:, 1:, 1:])

        pattern_filter = lambda name: "hook_pattern" in name
        self.model.run_with_hooks(
            text,
            return_type=None,
            fwd_hooks=[(pattern_filter, save_pattern)],
        )

        return {
            "tokens": tokens_list[1:],
            "patterns": patterns,
        }

    def process_text(self, text: str) -> Dict[str, Any]:
        result = self.get_attention_patterns(text)
        tokens = result["tokens"]
        patterns = result["patterns"]

        attention_patterns: List[Dict[str, Any]] = []
        for layer, layer_pattern in enumerate(patterns):
            for head in range(self.n_heads):
                head_pattern = layer_pattern[head]
                for src_idx in range(len(tokens)):
                    for dest_idx in range(len(tokens)):
                        weight = float(head_pattern[dest_idx, src_idx])
                        attention_patterns.append(
                            {
                                "sourceLayer": layer,
                                "sourceToken": src_idx,
                                "destLayer": layer + 1,
                                "destToken": dest_idx,
                                "weight": weight,
                                "head": head,
                            }
                        )

        return {
            "numLayers": self.n_layers + 1,
            "numTokens": len(tokens),
            "numHeads": self.n_heads,
            "tokens": tokens,
            "attentionPatterns": attention_patterns,
            "model_name": self.model_name,
            "model_info": {
                "name": self.model_name,
                "layers": self.n_layers,
                "heads": self.n_heads,
            },
        }

    def _resolve_target_token_id(self, target_token: Optional[str]) -> Optional[int]:
        if not target_token or not target_token.strip():
            return None

        candidates = self.model.to_tokens(target_token, prepend_bos=False).squeeze(0)
        if candidates.numel() == 0:
            return None
        return int(candidates[0].item())

    def _head_ablation_hook(self, ablated_heads: set[Tuple[int, int]]):
        def hook_fn(z, hook):
            layer = hook.layer()
            ablated_in_layer = [head for candidate_layer, head in ablated_heads if candidate_layer == layer]
            if not ablated_in_layer:
                return z
            z = z.clone()
            z[:, :, ablated_in_layer, :] = 0.0
            return z

        return hook_fn

    def _run_logits(self, text: str, ablated_heads: Optional[List[Tuple[int, int]]] = None) -> torch.Tensor:
        if not ablated_heads:
            return self.model(text, return_type="logits")

        hook_filter = lambda name: name.endswith("attn.hook_z")
        return self.model.run_with_hooks(
            text,
            return_type="logits",
            fwd_hooks=[(hook_filter, self._head_ablation_hook(set(ablated_heads)))],
        )

    def _loss_from_logits(self, logits: torch.Tensor, tokens: torch.Tensor) -> float:
        if logits.shape[1] < 2:
            return 0.0

        shifted_logits = logits[:, :-1, :]
        shifted_targets = tokens[:, 1:]
        loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            shifted_targets.reshape(-1),
        )
        return float(loss.item())

    def _build_metrics(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        target_token_id: Optional[int],
        baseline_target_logit: Optional[float] = None,
    ) -> Dict[str, Any]:
        final_logits = logits[0, -1]
        probs = F.softmax(final_logits, dim=-1)
        top_prob, top_id = torch.max(probs, dim=-1)
        top_token = self.model.to_string(torch.tensor([int(top_id.item())], device=logits.device))

        target_prob = None
        target_logit = None
        if target_token_id is not None:
            target_prob = float(probs[target_token_id].item())
            target_logit = float(final_logits[target_token_id].item())

        metrics: Dict[str, Any] = {
            "top_prediction": top_token,
            "top_probability": float(top_prob.item()),
            "loss": self._loss_from_logits(logits, tokens),
            "target_token_id": target_token_id,
            "target_probability": target_prob,
            "target_logit": target_logit,
        }

        if baseline_target_logit is not None and target_logit is not None:
            top_logit = float(final_logits[top_id].item())
            metrics["logit_diff"] = target_logit - top_logit
            metrics["logit_diff_delta"] = target_logit - baseline_target_logit
        elif target_logit is not None:
            competitor_index = int(torch.topk(final_logits, k=2).indices[1].item()) if final_logits.shape[0] > 1 else int(top_id.item())
            competitor_logit = float(final_logits[competitor_index].item())
            metrics["logit_diff"] = target_logit - competitor_logit
            metrics["logit_diff_delta"] = 0.0
        else:
            metrics["logit_diff"] = None
            metrics["logit_diff_delta"] = None

        top_k = torch.topk(probs, k=min(5, probs.shape[0]))
        metrics["top_tokens"] = [
            {
                "token": self.model.to_string(torch.tensor([int(token_id.item())], device=logits.device)),
                "probability": float(prob.item()),
                "token_id": int(token_id.item()),
            }
            for prob, token_id in zip(top_k.values, top_k.indices)
        ]
        return metrics

    def evaluate_text(
        self,
        text: str,
        target_token: Optional[str] = None,
        ablated_heads: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        token_tensor = self.model.to_tokens(text)
        token_strings = self.model.to_str_tokens(text)
        target_token_id = self._resolve_target_token_id(target_token)

        baseline_logits = self._run_logits(text)
        baseline_metrics = self._build_metrics(baseline_logits, token_tensor, target_token_id)

        result: Dict[str, Any] = {
            "tokens": token_strings,
            "target_token": target_token,
            "baseline": baseline_metrics,
            "ablated": None,
            "delta": None,
        }

        if ablated_heads:
            ablated_logits = self._run_logits(text, ablated_heads=ablated_heads)
            ablated_metrics = self._build_metrics(
                ablated_logits,
                token_tensor,
                target_token_id,
                baseline_target_logit=baseline_metrics["target_logit"],
            )
            result["ablated"] = ablated_metrics
            result["delta"] = {
                "target_probability_delta": (
                    None
                    if baseline_metrics["target_probability"] is None or ablated_metrics["target_probability"] is None
                    else ablated_metrics["target_probability"] - baseline_metrics["target_probability"]
                ),
                "loss_delta": ablated_metrics["loss"] - baseline_metrics["loss"],
                "logit_diff_delta": ablated_metrics["logit_diff_delta"],
                "top_prediction_changed": ablated_metrics["top_prediction"] != baseline_metrics["top_prediction"],
            }

        return result


if __name__ == "__main__":
    extractor = AttentionPatternExtractor()
    text = "The quick brown fox jumped over the lazy dog"
    patterns = extractor.process_text(text)
    print(f"Number of tokens: {patterns['numTokens']}")
    print(f"Number of attention patterns: {len(patterns['attentionPatterns'])}")
    print(f"Tokens: {patterns['tokens']}")
