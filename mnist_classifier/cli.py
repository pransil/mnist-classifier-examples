"""Command-line interface for MNIST Classifier."""

import click
import uvicorn
from pathlib import Path


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """MNIST Classifier - ML system for digit classification."""
    pass


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind the server to')
@click.option('--port', default=8000, help='Port to bind the server to')
@click.option('--models-dir', default='models', help='Directory containing saved models')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.option('--log-level', default='info', help='Log level (debug, info, warning, error)')
def serve(host, port, models_dir, reload, log_level):
    """Start the API server."""
    click.echo(f"🚀 Starting MNIST Classifier API server...")
    click.echo(f"   Host: {host}")
    click.echo(f"   Port: {port}")
    click.echo(f"   Models directory: {models_dir}")
    click.echo(f"   Reload: {reload}")
    
    # Set models directory as environment variable for the app
    import os
    os.environ['MODELS_DIR'] = models_dir
    
    uvicorn.run(
        "mnist_classifier.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--model', default='best', help='Model to use for prediction')
@click.option('--probabilities', is_flag=True, help='Show class probabilities')
def predict(image_path, model, probabilities):
    """Make a prediction on an image file."""
    from PIL import Image
    import numpy as np
    from .api.predictor import ModelPredictor
    
    click.echo(f"📸 Loading image: {image_path}")
    
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        img_array = np.array(img) / 255.0  # Normalize to 0-1
        
        # Initialize predictor
        predictor = ModelPredictor()
        
        # Make prediction
        predicted_digit, confidence, probs, model_used, processing_time = predictor.predict(
            img_array, model, probabilities
        )
        
        click.echo(f"🎯 Prediction Results:")
        click.echo(f"   Predicted digit: {predicted_digit}")
        click.echo(f"   Confidence: {confidence:.4f}")
        click.echo(f"   Model used: {model_used}")
        click.echo(f"   Processing time: {processing_time:.2f}ms")
        
        if probabilities and probs:
            click.echo(f"📊 Class probabilities:")
            for digit, prob in probs.items():
                click.echo(f"   {digit}: {prob:.4f}")
                
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--models-dir', default='models', help='Directory containing saved models')
def list_models(models_dir):
    """List available models."""
    from .api.predictor import ModelPredictor
    
    click.echo("🤖 Available Models:")
    
    try:
        predictor = ModelPredictor(models_dir)
        models_info = predictor.get_available_models()
        
        if not models_info:
            click.echo("   No models found.")
            return
        
        for model in models_info:
            click.echo(f"   📦 {model['name']}")
            click.echo(f"      Type: {model['type']}")
            if model.get('accuracy'):
                click.echo(f"      Accuracy: {model['accuracy']:.4f}")
            if model.get('parameters'):
                click.echo(f"      Parameters: {model['parameters']:,}")
            click.echo()
        
        click.echo(f"🏆 Best model: {predictor.get_best_model()}")
        click.echo(f"⚙️  Default model: {predictor.get_default_model()}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output-dir', default='results', help='Directory to save results')
def train(config_file, output_dir):
    """Train models using configuration file."""
    click.echo(f"🏋️ Starting training with config: {config_file}")
    click.echo(f"📁 Output directory: {output_dir}")
    
    # This would integrate with the training system
    click.echo("⚠️  Training integration not implemented yet.")
    click.echo("💡 Use the training scripts directly for now.")


@cli.command()
@click.option('--models-dir', default='models', help='Directory containing saved models') 
@click.option('--output-dir', default='reports', help='Directory to save evaluation report')
@click.option('--include-plots', is_flag=True, help='Generate evaluation plots')
def evaluate(models_dir, output_dir, include_plots):
    """Evaluate all available models."""
    click.echo(f"📊 Evaluating models from: {models_dir}")
    click.echo(f"📁 Output directory: {output_dir}")
    
    # This would integrate with the evaluation system
    click.echo("⚠️  Evaluation integration not implemented yet.")
    click.echo("💡 Use the evaluation scripts directly for now.")


if __name__ == '__main__':
    cli()