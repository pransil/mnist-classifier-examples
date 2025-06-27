"""Simple test script for the MNIST Classifier API."""

import json
import numpy as np
import requests
from typing import Dict, Any


def test_api_locally():
    """Test the API with sample data."""
    base_url = "http://localhost:8000/api/v1"
    
    print("🧪 Testing MNIST Classifier API")
    print("=" * 50)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check passed")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Models loaded: {health_data.get('models_loaded')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to API. Make sure it's running on localhost:8000")
        return
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return
    
    print()
    
    # Test models endpoint
    print("2. Testing models endpoint...")
    try:
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            models_data = response.json()
            print(f"   ✅ Models endpoint working")
            print(f"   Available models: {len(models_data.get('models', []))}")
            print(f"   Default model: {models_data.get('default_model')}")
            print(f"   Best model: {models_data.get('best_model')}")
        else:
            print(f"   ❌ Models endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Models endpoint error: {e}")
    
    print()
    
    # Test prediction endpoint
    print("3. Testing prediction endpoint...")
    try:
        # Create a simple test image (representing a "0")
        test_image = create_test_digit_0()
        
        prediction_request = {
            "image_data": test_image.tolist(),
            "model_name": "best",
            "return_probabilities": True
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=prediction_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            prediction_data = response.json()
            print(f"   ✅ Prediction successful")
            print(f"   Predicted digit: {prediction_data.get('predicted_digit')}")
            print(f"   Confidence: {prediction_data.get('confidence'):.4f}")
            print(f"   Model used: {prediction_data.get('model_used')}")
            print(f"   Processing time: {prediction_data.get('processing_time_ms'):.2f}ms")
            
            if prediction_data.get('probabilities'):
                print("   Top 3 probabilities:")
                probs = prediction_data['probabilities']
                sorted_probs = sorted(probs.items(), key=lambda x: float(x[1]), reverse=True)
                for digit, prob in sorted_probs[:3]:
                    print(f"     {digit}: {float(prob):.4f}")
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
    
    print()
    
    # Test batch prediction
    print("4. Testing batch prediction endpoint...")
    try:
        # Create multiple test images
        test_images = [
            create_test_digit_0().tolist(),
            create_test_digit_1().tolist(),
            create_test_digit_0().tolist()
        ]
        
        batch_request = {
            "images": test_images,
            "model_name": "best",
            "return_probabilities": False
        }
        
        response = requests.post(
            f"{base_url}/predict/batch",
            json=batch_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            batch_data = response.json()
            print(f"   ✅ Batch prediction successful")
            print(f"   Batch size: {batch_data.get('batch_size')}")
            print(f"   Total time: {batch_data.get('total_processing_time_ms'):.2f}ms")
            print(f"   Average per image: {batch_data.get('average_time_per_image_ms'):.2f}ms")
            
            predictions = batch_data.get('predictions', [])
            print("   Predictions:")
            for i, pred in enumerate(predictions):
                print(f"     Image {i}: {pred.get('predicted_digit')} (confidence: {pred.get('confidence'):.4f})")
        else:
            print(f"   ❌ Batch prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Batch prediction error: {e}")
    
    print()
    print("🎉 API testing completed!")


def create_test_digit_0() -> np.ndarray:
    """Create a simple test image that looks like a '0'."""
    img = np.zeros((28, 28), dtype=np.float32)
    
    # Draw a simple circle/oval for '0'
    center_x, center_y = 14, 14
    for i in range(28):
        for j in range(28):
            # Create an oval shape
            dx = (i - center_x) / 8.0
            dy = (j - center_y) / 6.0
            distance = dx * dx + dy * dy
            
            # Ring shape (hollow circle)
            if 0.7 < distance < 1.3:
                img[i, j] = 0.8
            elif 0.5 < distance < 1.5:
                img[i, j] = 0.4
    
    return img


def create_test_digit_1() -> np.ndarray:
    """Create a simple test image that looks like a '1'."""
    img = np.zeros((28, 28), dtype=np.float32)
    
    # Draw a simple vertical line for '1'
    center_x = 14
    for i in range(28):
        for j in range(6, 22):  # Vertical line
            if abs(j - center_x) <= 1:
                img[i, j] = 0.8
            elif abs(j - center_x) <= 2:
                img[i, j] = 0.4
    
    # Add a small diagonal line at the top
    for i in range(5, 10):
        j = center_x - (10 - i)
        if 0 <= j < 28:
            img[i, j] = 0.6
    
    return img


def generate_api_examples():
    """Generate example requests for documentation."""
    print("📋 API Examples")
    print("=" * 50)
    
    # Single prediction example
    test_image = create_test_digit_0()
    single_request = {
        "image_data": test_image.tolist(),
        "model_name": "cnn",
        "return_probabilities": True
    }
    
    print("Single Prediction Request:")
    print("POST /api/v1/predict")
    print("Content-Type: application/json")
    print()
    print(json.dumps(single_request, indent=2)[:500] + "...")
    print()
    
    # Batch prediction example
    batch_request = {
        "images": [
            create_test_digit_0().tolist(),
            create_test_digit_1().tolist()
        ],
        "model_name": "best",
        "return_probabilities": False
    }
    
    print("Batch Prediction Request:")
    print("POST /api/v1/predict/batch")
    print("Content-Type: application/json")
    print()
    print("{\n  \"images\": [\n    [[28x28 array]], \n    [[28x28 array]]\n  ],")
    print("  \"model_name\": \"best\",")
    print("  \"return_probabilities\": false")
    print("}")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        generate_api_examples()
    else:
        test_api_locally()