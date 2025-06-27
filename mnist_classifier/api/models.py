"""Pydantic models for MNIST Classifier API."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    
    image_data: List[List[float]] = Field(
        ..., 
        description="28x28 image data as nested list of floats (0-1 normalized)",
        min_items=28,
        max_items=28
    )
    
    model_name: Optional[str] = Field(
        default="best",
        description="Model to use for prediction (mlp, cnn, xgboost, or 'best')"
    )
    
    return_probabilities: bool = Field(
        default=False,
        description="Whether to return class probabilities"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_data": [[0.0] * 28 for _ in range(28)],
                "model_name": "cnn",
                "return_probabilities": True
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    predicted_digit: int = Field(
        ...,
        description="Predicted digit (0-9)",
        ge=0,
        le=9
    )
    
    confidence: float = Field(
        ...,
        description="Prediction confidence (0-1)",
        ge=0.0,
        le=1.0
    )
    
    probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Class probabilities for all digits"
    )
    
    model_used: str = Field(
        ...,
        description="Name of the model used for prediction"
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )


class ModelInfo(BaseModel):
    """Information about available models."""
    
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type (mlp, cnn, xgboost)")
    accuracy: Optional[float] = Field(default=None, description="Test accuracy")
    parameters: Optional[int] = Field(default=None, description="Number of parameters")
    training_time: Optional[float] = Field(default=None, description="Training time in seconds")
    is_available: bool = Field(..., description="Whether model is loaded and available")


class ModelsResponse(BaseModel):
    """Response model for available models."""
    
    models: List[ModelInfo] = Field(..., description="List of available models")
    default_model: str = Field(..., description="Default model name")
    best_model: str = Field(..., description="Best performing model")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: int = Field(..., description="Number of models loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    images: List[List[List[float]]] = Field(
        ...,
        description="List of 28x28 images as nested lists",
        min_items=1,
        max_items=100  # Limit batch size
    )
    
    model_name: Optional[str] = Field(
        default="best",
        description="Model to use for predictions"
    )
    
    return_probabilities: bool = Field(
        default=False,
        description="Whether to return class probabilities"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions"
    )
    
    batch_size: int = Field(
        ...,
        description="Number of images processed"
    )
    
    total_processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds"
    )
    
    average_time_per_image_ms: float = Field(
        ...,
        description="Average processing time per image"
    )