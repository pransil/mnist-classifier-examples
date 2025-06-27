"""Main FastAPI application for MNIST Classifier."""

import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .routes import router
from .predictor import ModelPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(models_dir: str = "models") -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        models_dir: Directory containing saved models
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="MNIST Classifier API",
        description="REST API for MNIST digit classification using multiple model types",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure as needed for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router, prefix="/api/v1", tags=["predictions"])
    
    # Initialize predictor with custom models directory
    if hasattr(router, 'predictor'):
        router.predictor = ModelPredictor(models_dir)
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Root endpoint with API information."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MNIST Classifier API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; }
                .endpoint { background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .method { font-weight: bold; color: #2563eb; }
                .path { font-family: monospace; background-color: #e5e7eb; padding: 2px 4px; border-radius: 3px; }
                a { color: #2563eb; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 MNIST Classifier API</h1>
                <p>Welcome to the MNIST digit classification API. This service provides predictions using trained machine learning models.</p>
                
                <h2>📚 Documentation</h2>
                <ul>
                    <li><a href="/docs">Interactive API Documentation (Swagger UI)</a></li>
                    <li><a href="/redoc">Alternative Documentation (ReDoc)</a></li>
                </ul>
                
                <h2>🔗 Available Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method">GET</span> <span class="path">/api/v1/health</span><br>
                    Check service health and status
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <span class="path">/api/v1/models</span><br>
                    Get information about available models
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <span class="path">/api/v1/predict</span><br>
                    Make a prediction on a single 28x28 image
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <span class="path">/api/v1/predict/batch</span><br>
                    Make predictions on multiple images
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <span class="path">/api/v1/models/{model_name}</span><br>
                    Get detailed information about a specific model
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <span class="path">/api/v1/models/{model_name}/predict</span><br>
                    Make a prediction using a specific model
                </div>
                
                <h2>🚀 Quick Start</h2>
                <p>To make a prediction, send a POST request to <code>/api/v1/predict</code> with a JSON payload:</p>
                <pre><code>{
  "image_data": [[0.0, 0.1, ...], [0.2, 0.3, ...], ...],  // 28x28 array
  "model_name": "best",  // optional
  "return_probabilities": false  // optional
}</code></pre>
                
                <h2>🔧 Model Types</h2>
                <ul>
                    <li><strong>MLP</strong>: Multi-Layer Perceptron with fully connected layers</li>
                    <li><strong>CNN</strong>: Convolutional Neural Network optimized for images</li>
                    <li><strong>XGBoost</strong>: Gradient boosting classifier</li>
                </ul>
                
                <p>For detailed usage examples, please visit the <a href="/docs">interactive documentation</a>.</p>
            </div>
        </body>
        </html>
        """
        return html_content
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize application on startup."""
        logger.info("Starting MNIST Classifier API...")
        
        # Verify models directory exists
        models_path = Path(models_dir)
        if not models_path.exists():
            logger.warning(f"Models directory {models_path} does not exist")
        else:
            logger.info(f"Using models directory: {models_path}")
        
        logger.info("MNIST Classifier API started successfully")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("Shutting down MNIST Classifier API...")
    
    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )