from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from .batch_eval import evaluate_dataset
from .datasets import load_dataset, list_datasets, save_custom_dataset, validate_dataset_payload

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AVAILABLE_MODELS = {
    "gpt2-small": "gpt2-small",
    "pythia-2.8b": "pythia-2.8b",
}

# Initialize FastAPI app
app = FastAPI(
    title="Attention Pattern API",
    description="API for extracting attention patterns from transformer models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a model registry to hold model instances
model_registry = {}

class HeadSelection(BaseModel):
    layer: int
    head: int


class TextRequest(BaseModel):
    text: str
    model_name: Optional[str] = "gpt2-small"
    threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    top_k: int = Field(default=12, ge=1, le=500)
    selected_heads: List[HeadSelection] = Field(default_factory=list)


class EvaluationRequest(BaseModel):
    text: str
    model_name: Optional[str] = "gpt2-small"
    target_token: Optional[str] = None
    ablated_heads: List[HeadSelection] = Field(default_factory=list)


class MaxLogitDiffGraphRequest(BaseModel):
    text: str
    corrupted_text: str
    model_name: Optional[str] = "gpt2-small"
    target_token: Optional[str] = None
    top_k: int = Field(default=12, ge=1, le=500)
    top_heads_per_layer: int = Field(default=3, ge=1, le=16)
    selected_heads: List[HeadSelection] = Field(default_factory=list)


class DatasetPayloadRequest(BaseModel):
    payload: Dict[str, Any]


class DatasetEvaluationRequest(BaseModel):
    dataset_name: str
    model_name: Optional[str] = None
    ablated_heads: List[HeadSelection] = Field(default_factory=list)


def get_or_load_model(model_name: str):
    """Load a model on demand so the API can bind its port before heavy startup work."""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not supported. Available models: {', '.join(AVAILABLE_MODELS.keys())}",
        )

    if model_name not in model_registry:
        try:
            from .model import AttentionPatternExtractor

            logger.info(f"Loading model '{model_name}'...")
            model_registry[model_name] = AttentionPatternExtractor(model_name)
            logger.info(f"Model '{model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}': {str(e)}")

    return model_registry[model_name]

@app.post("/process")
async def process_text(request: TextRequest) -> Dict[str, Any]:
    """Process text and return attention patterns.
    
    Args:
        request: TextRequest object containing the text to analyze and optional model name
        
    Returns:
        Dictionary containing:
            - numLayers: number of layers
            - numTokens: number of tokens
            - numHeads: number of attention heads
            - tokens: list of tokens
            - attentionHeads: grouped sparse attention edges
    """
    model_name = request.model_name
    
    model = get_or_load_model(model_name)
    
    try:
        logger.info(f"Processing text with model '{model_name}': {request.text[:50]}...")
        result = model.process_text(
            request.text,
            threshold=request.threshold,
            top_k=request.top_k,
            selected_heads=[(selection.layer, selection.head) for selection in request.selected_heads],
        )
        result["model_name"] = model_name  # Add model name to response
        logger.info(
            "Successfully processed text. Generated %s grouped heads with %s edges",
            len(result["attentionHeads"]),
            result["numEdges"],
        )
        return result
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate_text(request: EvaluationRequest) -> Dict[str, Any]:
    model_name = request.model_name

    model = get_or_load_model(model_name)

    try:
        logger.info(
            "Evaluating text with model '%s' and %s ablated heads",
            model_name,
            len(request.ablated_heads),
        )
        result = model.evaluate_text(
            text=request.text,
            target_token=request.target_token,
            ablated_heads=[(selection.layer, selection.head) for selection in request.ablated_heads],
        )
        result["model_name"] = model_name
        return result
    except Exception as e:
        logger.error(f"Error evaluating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/max-logit-diff-graph")
async def max_logit_diff_graph(request: MaxLogitDiffGraphRequest) -> Dict[str, Any]:
    model_name = request.model_name
    model = get_or_load_model(model_name)

    try:
        logger.info(
            "Building max-logit-diff graph for model '%s' across all layers at the final position",
            model_name,
        )
        result = model.build_max_logit_diff_graph(
            text=request.text,
            corrupted_text=request.corrupted_text,
            target_token=request.target_token,
            top_k=request.top_k,
            top_heads_per_layer=request.top_heads_per_layer,
            selected_heads=[(selection.layer, selection.head) for selection in request.selected_heads],
        )
        result["model_name"] = model_name
        return result
    except Exception as e:
        logger.error(f"Error building max-logit-diff graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets")
async def get_datasets() -> Dict[str, Any]:
    return {"datasets": list_datasets()}


@app.get("/datasets/{dataset_name}")
async def get_dataset(dataset_name: str) -> Dict[str, Any]:
    try:
        return load_dataset(dataset_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/datasets/validate")
async def validate_dataset(request: DatasetPayloadRequest) -> Dict[str, Any]:
    return validate_dataset_payload(request.payload)


@app.post("/datasets/save")
async def save_dataset(request: DatasetPayloadRequest) -> Dict[str, Any]:
    validation = validate_dataset_payload(request.payload)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=validation)
    try:
        return save_custom_dataset(request.payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/evaluate-dataset")
async def evaluate_dataset_endpoint(request: DatasetEvaluationRequest) -> Dict[str, Any]:
    try:
        dataset = load_dataset(request.dataset_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    model_name = request.model_name or dataset.get("model") or "gpt2-small"
    model = get_or_load_model(model_name)

    try:
        result = evaluate_dataset(
            model=model,
            dataset=dataset,
            ablated_heads=[(selection.layer, selection.head) for selection in request.ablated_heads],
        )
        result["model_name"] = model_name
        return result
    except Exception as exc:
        logger.error(f"Error evaluating dataset: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": list(AVAILABLE_MODELS.keys()),
        "loaded_models": list(model_registry.keys())
    }

@app.get("/")
async def root():
    """Minimal root endpoint for platform health checks."""
    return {
        "status": "ok",
        "service": "attention-api",
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "loaded_models": list(model_registry.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
