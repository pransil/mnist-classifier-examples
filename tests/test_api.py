"""Tests for MNIST Classifier API."""

import pytest
import json
import numpy as np
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from pathlib import Path

from mnist_classifier.api.main import create_app
from mnist_classifier.api.models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    ModelsResponse,
    HealthResponse
)
from mnist_classifier.api.predictor import ModelPredictor


class TestAPIModels:
    """Test Pydantic models for API."""
    
    def test_prediction_request_validation(self):
        """Test PredictionRequest validation."""
        # Valid request
        valid_data = {
            "image_data": [[0.1] * 28 for _ in range(28)],
            "model_name": "cnn",
            "return_probabilities": True
        }
        
        request = PredictionRequest(**valid_data)
        assert len(request.image_data) == 28
        assert len(request.image_data[0]) == 28
        assert request.model_name == "cnn"
        assert request.return_probabilities == True
    
    def test_prediction_request_invalid_size(self):
        """Test PredictionRequest with invalid image size."""
        # Wrong number of rows
        with pytest.raises(ValueError):
            PredictionRequest(
                image_data=[[0.1] * 28 for _ in range(30)],  # 30 rows instead of 28
                model_name="cnn"
            )
    
    def test_prediction_response_validation(self):
        """Test PredictionResponse validation."""
        response_data = {
            "predicted_digit": 7,
            "confidence": 0.95,
            "probabilities": {str(i): 0.1 for i in range(10)},
            "model_used": "cnn_medium",
            "processing_time_ms": 2.5
        }
        
        response = PredictionResponse(**response_data)
        assert response.predicted_digit == 7
        assert response.confidence == 0.95
        assert len(response.probabilities) == 10
    
    def test_batch_prediction_request(self):
        """Test BatchPredictionRequest validation."""
        batch_data = {
            "images": [
                [[0.1] * 28 for _ in range(28)],
                [[0.2] * 28 for _ in range(28)]
            ],
            "model_name": "best",
            "return_probabilities": False
        }
        
        request = BatchPredictionRequest(**batch_data)
        assert len(request.images) == 2
        assert request.model_name == "best"
    
    def test_batch_request_size_limit(self):
        """Test batch request size limitation."""
        # Create batch that's too large
        large_batch = {
            "images": [[[0.1] * 28 for _ in range(28)] for _ in range(150)],  # 150 images
            "model_name": "best"
        }
        
        with pytest.raises(ValueError):
            BatchPredictionRequest(**large_batch)


class TestModelPredictor:
    """Test ModelPredictor functionality."""
    
    def test_predictor_initialization(self, temp_models_dir):
        """Test predictor initialization."""
        predictor = ModelPredictor(models_dir=temp_models_dir)
        
        assert predictor.models_dir == Path(temp_models_dir)
        assert isinstance(predictor.models, dict)
        assert isinstance(predictor.model_info, dict)
    
    def test_predictor_with_no_models(self, temp_models_dir):
        """Test predictor with empty models directory."""
        predictor = ModelPredictor(models_dir=temp_models_dir)
        
        # Should handle gracefully
        assert len(predictor.models) == 0
        assert predictor.best_model_name is None
        
        models_info = predictor.get_available_models()
        assert len(models_info) == 0
    
    @patch('mnist_classifier.api.predictor.torch')
    @patch('mnist_classifier.api.predictor.TORCH_AVAILABLE', True)
    def test_pytorch_model_loading(self, mock_torch, temp_models_dir):
        """Test PyTorch model loading."""
        # Create mock model file
        model_file = Path(temp_models_dir) / "mlp_medium.pth"
        model_file.touch()
        
        # Mock torch.load to return a simple state dict
        mock_torch.load.return_value = {
            'model_state_dict': {'layer.weight': mock_torch.randn(10, 784)},
            'accuracy': 0.95
        }
        
        predictor = ModelPredictor(models_dir=temp_models_dir)
        
        # Verify mock was called
        assert mock_torch.load.called
    
    def test_prediction_with_mock_model(self, temp_models_dir):
        """Test prediction with mock model."""
        predictor = ModelPredictor(models_dir=temp_models_dir)
        
        # Add a mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.1, 0.9] + [0.0] * 8])  # Predict class 1
        
        predictor.models['mock_model'] = mock_model
        predictor.model_info['mock_model'] = {
            'type': 'xgboost',
            'accuracy': 0.95,
            'is_available': True
        }
        predictor.best_model_name = 'mock_model'
        
        # Create test image
        test_image = np.random.rand(28, 28)
        
        # Mock the prediction method
        with patch.object(predictor, '_predict_xgboost') as mock_predict:
            mock_predict.return_value = (1, 0.9, {'0': 0.1, '1': 0.9}, 'mock_model', 2.5)
            
            result = predictor.predict(test_image, return_probabilities=True)
            predicted_digit, confidence, probabilities, model_used, processing_time = result
            
            assert predicted_digit == 1
            assert confidence == 0.9
            assert probabilities is not None
            assert model_used == 'mock_model'
    
    def test_get_available_models(self, temp_models_dir):
        """Test getting available models info."""
        predictor = ModelPredictor(models_dir=temp_models_dir)
        
        # Add mock models
        predictor.models['model1'] = Mock()
        predictor.models['model2'] = Mock()
        predictor.model_info['model1'] = {
            'type': 'mlp',
            'accuracy': 0.95,
            'is_available': True
        }
        predictor.model_info['model2'] = {
            'type': 'cnn', 
            'accuracy': 0.97,
            'is_available': True
        }
        
        models_info = predictor.get_available_models()
        
        assert len(models_info) == 2
        assert models_info[0]['name'] in ['model1', 'model2']
        assert all('type' in info for info in models_info)


class TestAPIEndpoints:
    """Test API endpoints."""
    
    @pytest.fixture
    def client(self, temp_models_dir):
        """Create test client."""
        app = create_app(models_dir=temp_models_dir)
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "MNIST Classifier API" in response.text
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        
        health_data = response.json()
        assert "status" in health_data
        assert "version" in health_data
        assert "models_loaded" in health_data
        assert health_data["status"] == "healthy"
    
    def test_models_endpoint(self, client):
        """Test models listing endpoint."""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        
        models_data = response.json()
        assert "models" in models_data
        assert "default_model" in models_data
        assert "best_model" in models_data
        assert isinstance(models_data["models"], list)
    
    def test_predict_endpoint_invalid_data(self, client):
        """Test prediction with invalid data."""
        # Invalid image size
        invalid_request = {
            "image_data": [[0.1] * 30 for _ in range(30)],  # Wrong size
            "model_name": "best"
        }
        
        response = client.post("/api/v1/predict", json=invalid_request)
        
        assert response.status_code == 400
        assert "28x28" in response.json()["detail"]
    
    def test_predict_endpoint_invalid_pixel_values(self, client):
        """Test prediction with invalid pixel values."""
        # Pixel values out of range
        invalid_request = {
            "image_data": [[1.5] * 28 for _ in range(28)],  # Values > 1
            "model_name": "best"
        }
        
        response = client.post("/api/v1/predict", json=invalid_request)
        
        assert response.status_code == 400
        assert "between 0 and 1" in response.json()["detail"]
    
    @patch('mnist_classifier.api.routes.predictor')
    def test_predict_endpoint_success(self, mock_predictor, client):
        """Test successful prediction."""
        # Mock predictor response
        mock_predictor.predict.return_value = (7, 0.95, None, "mock_model", 2.5)
        
        valid_request = {
            "image_data": [[0.1] * 28 for _ in range(28)],
            "model_name": "best",
            "return_probabilities": False
        }
        
        response = client.post("/api/v1/predict", json=valid_request)
        
        assert response.status_code == 200
        
        prediction_data = response.json()
        assert prediction_data["predicted_digit"] == 7
        assert prediction_data["confidence"] == 0.95
        assert prediction_data["model_used"] == "mock_model"
        assert prediction_data["processing_time_ms"] == 2.5
    
    @patch('mnist_classifier.api.routes.predictor')
    def test_batch_predict_endpoint(self, mock_predictor, client):
        """Test batch prediction endpoint."""
        # Mock batch prediction response
        mock_predictor.batch_predict.return_value = [
            (1, 0.9, None, "mock_model", 2.0),
            (2, 0.85, None, "mock_model", 1.8)
        ]
        
        batch_request = {
            "images": [
                [[0.1] * 28 for _ in range(28)],
                [[0.2] * 28 for _ in range(28)]
            ],
            "model_name": "best",
            "return_probabilities": False
        }
        
        response = client.post("/api/v1/predict/batch", json=batch_request)
        
        assert response.status_code == 200
        
        batch_data = response.json()
        assert batch_data["batch_size"] == 2
        assert len(batch_data["predictions"]) == 2
        assert "total_processing_time_ms" in batch_data
        assert "average_time_per_image_ms" in batch_data
    
    def test_batch_predict_size_limit(self, client):
        """Test batch prediction size limit."""
        # Create batch that's too large
        large_batch = {
            "images": [[[0.1] * 28 for _ in range(28)] for _ in range(150)],
            "model_name": "best"
        }
        
        response = client.post("/api/v1/predict/batch", json=large_batch)
        
        assert response.status_code == 400
        assert "exceed 100" in response.json()["detail"]
    
    @patch('mnist_classifier.api.routes.predictor')
    def test_model_specific_prediction(self, mock_predictor, client):
        """Test model-specific prediction endpoint."""
        # Mock model availability
        mock_predictor.is_model_available.return_value = True
        mock_predictor.predict.return_value = (3, 0.88, None, "specific_model", 3.2)
        
        valid_request = {
            "image_data": [[0.1] * 28 for _ in range(28)],
            "return_probabilities": False
        }
        
        response = client.post("/api/v1/models/specific_model/predict", json=valid_request)
        
        assert response.status_code == 200
        
        prediction_data = response.json()
        assert prediction_data["predicted_digit"] == 3
        assert prediction_data["model_used"] == "specific_model"
    
    @patch('mnist_classifier.api.routes.predictor')
    def test_model_not_found(self, mock_predictor, client):
        """Test prediction with non-existent model."""
        # Mock model not available
        mock_predictor.is_model_available.return_value = False
        
        valid_request = {
            "image_data": [[0.1] * 28 for _ in range(28)]
        }
        
        response = client.post("/api/v1/models/nonexistent/predict", json=valid_request)
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @patch('mnist_classifier.api.routes.predictor')
    def test_get_model_info(self, mock_predictor, client):
        """Test getting specific model information."""
        # Mock model info
        mock_predictor.get_model_info.return_value = {
            'name': 'test_model',
            'type': 'cnn',
            'accuracy': 0.95,
            'parameters': 50000,
            'is_available': True
        }
        
        response = client.get("/api/v1/models/test_model")
        
        assert response.status_code == 200
        
        model_info = response.json()
        assert model_info["name"] == "test_model"
        assert model_info["type"] == "cnn"
        assert model_info["accuracy"] == 0.95
    
    @patch('mnist_classifier.api.routes.predictor')
    def test_get_model_info_not_found(self, mock_predictor, client):
        """Test getting info for non-existent model."""
        # Mock model not found
        mock_predictor.get_model_info.return_value = None
        
        response = client.get("/api/v1/models/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestAPIErrorHandling:
    """Test API error handling."""
    
    @pytest.fixture
    def client(self, temp_models_dir):
        """Create test client."""
        app = create_app(models_dir=temp_models_dir)
        return TestClient(app)
    
    def test_malformed_json(self, client):
        """Test handling of malformed JSON."""
        response = client.post(
            "/api/v1/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        incomplete_request = {
            "model_name": "best"
            # Missing image_data
        }
        
        response = client.post("/api/v1/predict", json=incomplete_request)
        
        assert response.status_code == 422
    
    @patch('mnist_classifier.api.routes.predictor')
    def test_prediction_internal_error(self, mock_predictor, client):
        """Test handling of internal prediction errors."""
        # Mock predictor to raise exception
        mock_predictor.predict.side_effect = Exception("Internal error")
        
        valid_request = {
            "image_data": [[0.1] * 28 for _ in range(28)],
            "model_name": "best"
        }
        
        response = client.post("/api/v1/predict", json=valid_request)
        
        assert response.status_code == 500
    
    def test_invalid_http_method(self, client):
        """Test invalid HTTP method."""
        response = client.put("/api/v1/predict")
        
        assert response.status_code == 405  # Method Not Allowed
    
    def test_invalid_endpoint(self, client):
        """Test accessing invalid endpoint."""
        response = client.get("/api/v1/invalid_endpoint")
        
        assert response.status_code == 404


class TestAPICORS:
    """Test CORS configuration."""
    
    @pytest.fixture
    def client(self, temp_models_dir):
        """Create test client."""
        app = create_app(models_dir=temp_models_dir)
        return TestClient(app)
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/v1/health")
        
        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    @pytest.fixture
    def client(self, temp_models_dir):
        """Create test client."""
        app = create_app(models_dir=temp_models_dir)
        return TestClient(app)
    
    def test_swagger_docs(self, client):
        """Test Swagger documentation endpoint."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_docs(self, client):
        """Test ReDoc documentation endpoint."""
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema endpoint."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check that our endpoints are documented
        assert "/api/v1/health" in schema["paths"]
        assert "/api/v1/predict" in schema["paths"]
        assert "/api/v1/models" in schema["paths"]