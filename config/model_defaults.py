"""Configuration for model-specific default values."""

MODEL_DEFAULTS = {
    "gpt2-small": {
        "default_text": "When Mary and John went the store, John gave a drink to",
        "task_presets": [
            {
                "name": "IOI",
                "text": "When Mary and John went the store, John gave a drink to",
                "target_token": " Mary",
                "comparison_text": "When Mary and John went the store, Mary gave a drink to",
                "description": "Indirect object identification prompt focused on name mover behavior."
            },
            {
                "name": "Induction",
                "text": "Alice saw a bright red kite and later Alice saw a bright red",
                "target_token": " kite",
                "comparison_text": "Alice saw a bright red kite and later Bob saw a bright red",
                "description": "Repeated-prefix prompt for induction-style copying."
            },
            {
                "name": "Factual Recall",
                "text": "The capital of France is",
                "target_token": " Paris",
                "comparison_text": "The capital of Italy is",
                "description": "Simple factual completion with a single expected answer token."
            }
        ]
    },
    "pythia-2.8b": {
        "default_text": "The quick brown fox jumps over the lazy dog",
        "task_presets": [
            {
                "name": "Induction",
                "text": "The river stone rolled past the gate and the river stone rolled past the",
                "target_token": " gate",
                "comparison_text": "The river stone rolled past the gate and the silver stone rolled past the",
                "description": "Repeated phrase continuation for induction behavior."
            },
            {
                "name": "Factual Recall",
                "text": "The capital city of France is",
                "target_token": " Paris",
                "comparison_text": "The capital city of Germany is",
                "description": "Single-token factual continuation."
            }
        ]
    }
}
