from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from .model import AttentionPatternExtractor, AVAILABLE_MODELS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize default model
try:
    default_model_name = "gpt2-small"
    model_registry[default_model_name] = AttentionPatternExtractor(default_model_name)
    logger.info(f"Default model '{default_model_name}' initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize default model: {str(e)}")
    raise

class TextRequest(BaseModel):
    text: str
    model_name: Optional[str] = "gpt2-small"


class HeadSelection(BaseModel):
    layer: int
    head: int


class EvaluationRequest(BaseModel):
    text: str
    model_name: Optional[str] = "gpt2-small"
    target_token: Optional[str] = None
    ablated_heads: List[HeadSelection] = Field(default_factory=list)

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
            - attentionPatterns: list of attention patterns
    """
    model_name = request.model_name
    
    # Validate model name
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not supported. Available models: {', '.join(AVAILABLE_MODELS.keys())}")
    
    # Load model if not already loaded
    if model_name not in model_registry:
        try:
            logger.info(f"Loading model '{model_name}'...")
            model_registry[model_name] = AttentionPatternExtractor(model_name)
            logger.info(f"Model '{model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}': {str(e)}")
    
    try:
        logger.info(f"Processing text with model '{model_name}': {request.text[:50]}...")
        result = model_registry[model_name].process_text(request.text)
        result["model_name"] = model_name  # Add model name to response
        logger.info(f"Successfully processed text. Generated {len(result['attentionPatterns'])} patterns")
        return result
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate_text(request: EvaluationRequest) -> Dict[str, Any]:
    model_name = request.model_name

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not supported. Available models: {', '.join(AVAILABLE_MODELS.keys())}",
        )

    if model_name not in model_registry:
        try:
            logger.info(f"Loading model '{model_name}'...")
            model_registry[model_name] = AttentionPatternExtractor(model_name)
            logger.info(f"Model '{model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}': {str(e)}")

    try:
        logger.info(
            "Evaluating text with model '%s' and %s ablated heads",
            model_name,
            len(request.ablated_heads),
        )
        result = model_registry[model_name].evaluate_text(
            text=request.text,
            target_token=request.target_token,
            ablated_heads=[(selection.layer, selection.head) for selection in request.ablated_heads],
        )
        result["model_name"] = model_name
        return result
    except Exception as e:
        logger.error(f"Error evaluating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": list(AVAILABLE_MODELS.keys()),
        "loaded_models": list(model_registry.keys())
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
