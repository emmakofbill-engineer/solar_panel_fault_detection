"""
Solar Panel Fault Detection - Inference Script
==============================================
Make predictions on thermal images
"""

from ultralytics import YOLO
import sys

def predict_image(image_path, model_path='runs/classify/solar_fault_detection/weights/best.pt'):
    """
    Predict fault type for a thermal image
    
    Args:
        image_path: Path to thermal image
        model_path: Path to trained model
    """
    
    print(f"🔮 Loading model from: {model_path}")
    model = YOLO(model_path)
    
    print(f"📸 Analyzing image: {image_path}")
    results = model.predict(
        source=image_path,
        save=True,
        conf=0.5,
    )
    
    # Get top prediction
    top_class = results[0].names[results[0].probs.top1]
    top_conf = results[0].probs.top1conf.item()
    
    print(f"\n✅ Prediction: {top_class}")
    print(f"   Confidence: {top_conf:.2%}")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py data/images/test/Cell/1234.jpg")
    else:
        predict_image(sys.argv[1])
