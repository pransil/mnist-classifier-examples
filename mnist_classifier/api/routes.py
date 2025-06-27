"""API routes for MNIST Classifier."""

import time
import logging
from typing import List
import numpy as np
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from .models import (
    PredictionRequest,
    PredictionResponse, 
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelsResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse
)
from .predictor import ModelPredictor

logger = logging.getLogger(__name__)

# Global predictor instance
predictor = ModelPredictor()

# Create router
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        start_time = getattr(health_check, 'start_time', time.time())
        uptime = time.time() - start_time
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            models_loaded=len(predictor.models),
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service unhealthy"
        )

# Store start time for uptime calculation
health_check.start_time = time.time()


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """Get information about available models."""
    try:
        models_info = predictor.get_available_models()
        
        return ModelsResponse(
            models=[ModelInfo(**model) for model in models_info],
            default_model=predictor.get_default_model(),
            best_model=predictor.get_best_model()
        )
    except Exception as e:
        logger.error(f"Failed to get models info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve models information"
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction on a single image."""
    try:
        # Validate image data format
        if len(request.image_data) != 28:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image data must be 28x28 pixels"
            )
        
        for row in request.image_data:
            if len(row) != 28:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Each row must contain exactly 28 pixels"
                )
        
        # Convert to numpy array
        image_array = np.array(request.image_data, dtype=np.float32)
        
        # Validate pixel values
        if np.any(image_array < 0) or np.any(image_array > 1):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Pixel values must be between 0 and 1"
            )
        
        # Make prediction
        predicted_digit, confidence, probabilities, model_used, processing_time = predictor.predict(
            image_array,
            request.model_name,
            request.return_probabilities
        )
        
        return PredictionResponse(
            predicted_digit=predicted_digit,
            confidence=confidence,
            probabilities=probabilities,
            model_used=model_used,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Prediction validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Make predictions on multiple images."""
    try:
        start_time = time.time()
        
        # Validate batch size
        if len(request.images) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size cannot exceed 100 images"
            )
        
        # Validate image format
        for i, image_data in enumerate(request.images):
            if len(image_data) != 28:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image {i} must be 28x28 pixels"
                )
            
            for j, row in enumerate(image_data):
                if len(row) != 28:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Image {i}, row {j} must contain exactly 28 pixels"
                    )
        
        # Convert to numpy arrays
        image_arrays = []
        for image_data in request.images:
            image_array = np.array(image_data, dtype=np.float32)
            
            # Validate pixel values
            if np.any(image_array < 0) or np.any(image_array > 1):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="All pixel values must be between 0 and 1"
                )
            
            image_arrays.append(image_array)
        
        # Make batch predictions
        results = predictor.batch_predict(
            image_arrays,
            request.model_name,
            request.return_probabilities
        )
        
        # Convert results to response format
        predictions = []
        for predicted_digit, confidence, probabilities, model_used, processing_time in results:
            predictions.append(PredictionResponse(
                predicted_digit=predicted_digit,
                confidence=confidence,
                probabilities=probabilities,
                model_used=model_used,
                processing_time_ms=processing_time
            ))
        
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(request.images)
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(request.images),
            total_processing_time_ms=total_time,
            average_time_per_image_ms=avg_time
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Batch prediction validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed"
        )


@router.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    try:
        model_info = predictor.get_model_info(model_name)
        
        if model_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        return ModelInfo(**model_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info for {model_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )


@router.post("/models/{model_name}/predict", response_model=PredictionResponse)
async def predict_with_model(model_name: str, request: PredictionRequest):
    """Make a prediction using a specific model."""
    # Override the model name from the path parameter
    request.model_name = model_name
    
    # Check if model exists
    if not predictor.is_model_available(model_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found or not available"
        )
    
    return await predict(request)


# Error handlers
@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail
        ).dict()
    )


@router.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred"
        ).dict()
    )