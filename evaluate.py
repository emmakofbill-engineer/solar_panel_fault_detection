"""
Solar Panel Fault Detection - Model Evaluation
===============================================
Evaluates trained YOLOv8 model on test dataset
"""

from ultralytics import YOLO

def evaluate_model(model_path='runs/classify/solar_fault_detection/weights/best.pt'):
    """Evaluate the trained model on test set"""
    
    print("🔍 Loading trained model...")
    model = YOLO(model_path)
    
    print("📊 Evaluating on test set...")
    metrics = model.val(
        data='data/images',
        split='test',
        batch=32,
        imgsz=224,
    )
    
    print("\n✅ Evaluation Results:")
    print(f"   Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"   Top-5 Accuracy: {metrics.top5:.4f}")
    
    return metrics

if __name__ == "__main__":
    print("="*70)
    print("MODEL EVALUATION - TEST SET")
    print("="*70)
    
    metrics = evaluate_model()
