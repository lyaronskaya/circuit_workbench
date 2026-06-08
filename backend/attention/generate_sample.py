from model import AttentionPatternExtractor, AVAILABLE_MODELS
import json
import os
import sys

def generate_sample_data(output_dir=None):
    """Generate sample attention pattern data for the available models.
    
    Args:
        output_dir: Optional directory to save the sample data. If None, will save to
                    '../../public/data' by default.
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        # First try to save directly to the public/data directory if we're running in the project root
        if os.path.exists("public/data"):
            output_dir = "public/data"
        else:
            # Fall back to a relative path from the backend directory
            output_dir = "../../public/data"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving sample data to: {os.path.abspath(output_dir)}")
    
    # Define model-specific sample texts
    sample_texts = {
        "gpt2-small": "When Mary and John went the store, John gave a drink to",
        "pythia-2.8b": "Fact: The Colosseum is in the country of"
    }
    
    # Generate sample data for each available model
    for model_name in AVAILABLE_MODELS:
        print(f"\nGenerating sample data for model: {model_name}")
        
        # Get the specific text for this model, or use a default if not specified
        text = sample_texts.get(model_name, "This is a sample text for testing attention patterns.")
        print(f"Using sample text: '{text}'")
        
        # Initialize the model with the specific model name
        try:
            extractor = AttentionPatternExtractor(model_name=model_name)
            
            # Generate attention patterns for sample text
            result = extractor.process_text(text)
            
            # Save to model-specific JSON file
            output_path = os.path.join(output_dir, f"sample-attention-{model_name}.json")
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            
            print(f"✅ Sample attention patterns saved to {output_path}")
            print(f"   Number of tokens: {result['numTokens']}")
            print(f"   Number of grouped heads: {len(result['attentionHeads'])}")
            print(f"   Number of attention edges: {result['numEdges']}")
            print(f"   Tokens: {result['tokens']}")
        except Exception as e:
            print(f"❌ Error generating sample data for {model_name}: {e}")

if __name__ == "__main__":
    # Allow specifying custom output directory as command line argument
    output_dir = sys.argv[1] if len(sys.argv) > 1 else None
    generate_sample_data(output_dir)
    
    print("\nSample data generation complete!")
    print("If running on the Jupyter server:")
    print("1. The files are saved in the sample_data directory")
    print("2. Download them from the Jupyter interface")
    print("3. Place them in your local public/data directory")
    print("4. Reload your frontend application") 
