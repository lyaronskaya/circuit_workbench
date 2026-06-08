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

    def _build_sparse_attention_heads(
        self,
        patterns: List[np.ndarray],
        num_tokens: int,
        threshold: float,
        top_k: int,
        selected_heads: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        if selected_heads is not None and len(selected_heads) == 0:
            return [], 0

        visible_heads = set(selected_heads or [])
        use_all_heads = selected_heads is None
        attention_heads: List[Dict[str, Any]] = []
        total_edges = 0

        for layer, layer_pattern in enumerate(patterns):
            for head in range(self.n_heads):
                if not use_all_heads and (layer, head) not in visible_heads:
                    continue

                head_pattern = layer_pattern[head]
                candidate_edges: List[Tuple[int, int, float]] = []
                for dest_idx in range(num_tokens):
                    for src_idx in range(num_tokens):
                        weight = float(head_pattern[dest_idx, src_idx])
                        if weight >= threshold:
                            candidate_edges.append((src_idx, dest_idx, weight))

                if not candidate_edges:
                    continue

                candidate_edges.sort(key=lambda edge: edge[2], reverse=True)
                limited_edges = candidate_edges[:top_k]
                total_edges += len(limited_edges)
                attention_heads.append(
                    {
                        "layer": layer,
                        "head": head,
                        "edges": [[src_idx, dest_idx, weight] for src_idx, dest_idx, weight in limited_edges],
                    }
                )

        return attention_heads, total_edges

    def process_text(
        self,
        text: str,
        threshold: float = 0.4,
        top_k: int = 12,
        selected_heads: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        result = self.get_attention_patterns(text)
        tokens = result["tokens"]
        patterns = result["patterns"]
        attention_heads, total_edges = self._build_sparse_attention_heads(
            patterns=patterns,
            num_tokens=len(tokens),
            threshold=threshold,
            top_k=top_k,
            selected_heads=selected_heads,
        )

        return {
            "numLayers": self.n_layers + 1,
            "numTokens": len(tokens),
            "numHeads": self.n_heads,
            "tokens": tokens,
            "attentionFormat": "grouped_sparse_v1",
            "attentionHeads": attention_heads,
            "numEdges": total_edges,
            "threshold": threshold,
            "top_k": top_k,
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

    def _top_non_target_token_id(self, logits: torch.Tensor, target_token_id: int) -> int:
        competitor_logits = logits.clone()
        competitor_logits[target_token_id] = float("-inf")
        return int(torch.argmax(competitor_logits).item())

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
        baseline_logit_diff: Optional[float] = None,
        baseline_target_rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        final_logits = logits[0, -1]
        probs = F.softmax(final_logits, dim=-1)
        top_prob, top_id = torch.max(probs, dim=-1)
        top_token = self.model.to_string(torch.tensor([int(top_id.item())], device=logits.device))

        target_prob = None
        target_logit = None
        target_rank = None
        logit_diff = None
        if target_token_id is not None:
            target_prob = float(probs[target_token_id].item())
            target_logit = float(final_logits[target_token_id].item())
            higher_logits = int((final_logits > final_logits[target_token_id]).sum().item())
            target_rank = higher_logits + 1

            if int(top_id.item()) == target_token_id:
                competitor_values, competitor_indices = torch.topk(final_logits, k=min(2, final_logits.shape[0]))
                competitor_logit = (
                    float(competitor_values[1].item())
                    if competitor_values.shape[0] > 1 else float(competitor_values[0].item())
                )
            else:
                competitor_logit = float(final_logits[top_id].item())
            logit_diff = target_logit - competitor_logit

        metrics: Dict[str, Any] = {
            "top_prediction": top_token,
            "top_probability": float(top_prob.item()),
            "loss": self._loss_from_logits(logits, tokens),
            "target_token_id": target_token_id,
            "target_probability": target_prob,
            "target_logit": target_logit,
            "target_rank": target_rank,
            "logit_diff": logit_diff,
        }

        if baseline_logit_diff is not None and logit_diff is not None:
            metrics["logit_diff_delta"] = logit_diff - baseline_logit_diff
        else:
            metrics["logit_diff_delta"] = 0.0 if logit_diff is not None else None

        if baseline_target_rank is not None and target_rank is not None:
            metrics["target_rank_delta"] = target_rank - baseline_target_rank
        else:
            metrics["target_rank_delta"] = 0 if target_rank is not None else None

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
                baseline_logit_diff=baseline_metrics["logit_diff"],
                baseline_target_rank=baseline_metrics["target_rank"],
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
                "target_rank_delta": ablated_metrics["target_rank_delta"],
                "top_prediction_changed": ablated_metrics["top_prediction"] != baseline_metrics["top_prediction"],
            }

        return result

    def build_max_logit_diff_graph(
        self,
        text: str,
        corrupted_text: str,
        target_token: Optional[str] = None,
        top_k: int = 12,
        top_heads_per_layer: int = 3,
        selected_heads: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        clean_visible_tokens = self.model.to_str_tokens(text)[1:]
        corrupted_visible_tokens = self.model.to_str_tokens(corrupted_text)[1:]

        if len(clean_visible_tokens) != len(corrupted_visible_tokens):
            raise ValueError("Clean and corrupted prompts must tokenize to the same visible token length.")

        clean_tokens = self.model.to_tokens(text)
        corrupted_tokens = self.model.to_tokens(corrupted_text)
        position = len(clean_visible_tokens) - 1
        model_position = position + 1

        if model_position >= clean_tokens.shape[1] or model_position >= corrupted_tokens.shape[1]:
            raise ValueError("Selected position is outside the model token range.")

        target_token_id = self._resolve_target_token_id(target_token)
        if target_token_id is None:
            raise ValueError("Target token is required when analyzing the final visible position.")

        pattern_names = {f"blocks.{layer}.attn.hook_pattern" for layer in range(self.n_layers)}
        z_names = {f"blocks.{layer}.attn.hook_z" for layer in range(self.n_layers)}
        cache_names = lambda name: name in pattern_names or name in z_names

        clean_logits, clean_cache = self.model.run_with_cache(
            text,
            return_type="logits",
            names_filter=cache_names,
        )
        corrupted_logits, corrupted_cache = self.model.run_with_cache(
            corrupted_text,
            return_type="logits",
            names_filter=cache_names,
        )

        prediction_index = model_position
        clean_final_logits = clean_logits[0, prediction_index]
        corrupted_final_logits = corrupted_logits[0, prediction_index]
        clean_competitor_id = self._top_non_target_token_id(clean_final_logits, target_token_id)
        corrupted_competitor_id = self._top_non_target_token_id(corrupted_final_logits, target_token_id)

        clean_direction = self.model.W_U[:, target_token_id] - self.model.W_U[:, clean_competitor_id]
        corrupted_direction = self.model.W_U[:, target_token_id] - self.model.W_U[:, corrupted_competitor_id]

        visible_heads_by_layer: Dict[int, set[int]] = {}
        if selected_heads:
            for candidate_layer, head in selected_heads:
                visible_heads_by_layer.setdefault(candidate_layer, set()).add(head)

        all_head_scores: List[Dict[str, Any]] = []
        important_nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        for layer in range(self.n_layers):
            pattern_name = f"blocks.{layer}.attn.hook_pattern"
            z_name = f"blocks.{layer}.attn.hook_z"
            clean_patterns = clean_cache[pattern_name][0]
            clean_z = clean_cache[z_name][0]
            corrupted_z = corrupted_cache[z_name][0]
            layer_w_o = self.model.W_O[layer]
            clean_results = torch.einsum("phd,hdm->phm", clean_z, layer_w_o)
            corrupted_results = torch.einsum("phd,hdm->phm", corrupted_z, layer_w_o)

            allowed_heads = visible_heads_by_layer.get(layer)
            use_all_heads = allowed_heads is None

            head_scores: List[Dict[str, Any]] = []
            for head in range(self.n_heads):
                if not use_all_heads and head not in allowed_heads:
                    continue

                clean_result = clean_results[model_position, head, :]
                corrupted_result = corrupted_results[model_position, head, :]
                clean_contribution = float(torch.dot(clean_result, clean_direction).item())
                corrupted_contribution = float(torch.dot(corrupted_result, corrupted_direction).item())
                head_scores.append(
                    {
                        "layer": layer,
                        "head": head,
                        "clean_logit_diff": clean_contribution,
                        "corrupted_logit_diff": corrupted_contribution,
                        "logit_diff_delta": clean_contribution - corrupted_contribution,
                    }
                )

            if not head_scores:
                continue

            all_head_scores.extend(head_scores)
            layer_important_heads = sorted(
                head_scores,
                key=lambda item: item["logit_diff_delta"],
                reverse=True,
            )[:top_heads_per_layer]

            for rank, important_head in enumerate(layer_important_heads):
                head_index = important_head["head"]
                attention_row = clean_patterns[head_index, model_position, 1:].detach().cpu().numpy()
                important_nodes.append(
                    {
                        "layer": layer + 1,
                        "token": position,
                        "sourceLayer": layer,
                        "head": head_index,
                        "rankInLayer": rank,
                        "logit_diff_delta": float(important_head["logit_diff_delta"]),
                        "clean_logit_diff": float(important_head["clean_logit_diff"]),
                        "corrupted_logit_diff": float(important_head["corrupted_logit_diff"]),
                        "clean_competitor": self.model.to_string(
                            torch.tensor([clean_competitor_id], device=self.device)
                        ),
                        "corrupted_competitor": self.model.to_string(
                            torch.tensor([corrupted_competitor_id], device=self.device)
                        ),
                    }
                )

                candidate_edges: List[Dict[str, Any]] = []
                for source_token, attention_weight in enumerate(attention_row.tolist()):
                    combined_score = float(attention_weight) * float(important_head["logit_diff_delta"])
                    candidate_edges.append(
                        {
                            "sourceLayer": layer,
                            "sourceToken": source_token,
                            "destToken": position,
                            "weight": float(attention_weight),
                            "head": head_index,
                            "rankInLayer": rank,
                            "logit_diff_delta": float(important_head["logit_diff_delta"]),
                            "clean_logit_diff": float(important_head["clean_logit_diff"]),
                            "corrupted_logit_diff": float(important_head["corrupted_logit_diff"]),
                            "combined_score": combined_score,
                        }
                    )

                candidate_edges.sort(key=lambda edge: abs(edge["combined_score"]), reverse=True)
                edges.extend(candidate_edges[:top_k])

        if not important_nodes:
            raise ValueError("No heads are available under the current filter.")

        return {
            "graphMode": "max_logit_diff_all_layers_last_position",
            "numLayers": self.n_layers + 1,
            "numTokens": len(clean_visible_tokens),
            "numHeads": self.n_heads,
            "tokens": clean_visible_tokens,
            "selectedPosition": position,
            "targetToken": self.model.to_string(torch.tensor([target_token_id], device=self.device)),
            "targetTokenId": target_token_id,
            "topHeadsPerLayer": top_heads_per_layer,
            "importantNodes": important_nodes,
            "headScores": all_head_scores,
            "attentionPatterns": edges,
            "numEdges": len(edges),
        }


if __name__ == "__main__":
    extractor = AttentionPatternExtractor()
    text = "The quick brown fox jumped over the lazy dog"
    patterns = extractor.process_text(text)
    print(f"Number of tokens: {patterns['numTokens']}")
    print(f"Number of grouped heads: {len(patterns['attentionHeads'])}")
    print(f"Number of attention edges: {patterns['numEdges']}")
    print(f"Tokens: {patterns['tokens']}")
