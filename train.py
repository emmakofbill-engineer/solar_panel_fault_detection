"""
Solar Panel Fault Detection - YOLOv8 Classification Training
==============================================================

Description:
    Trains a YOLOv8 classification model to detect faults in solar panels
    using thermal imagery. The model classifies 12 different fault types
    to enable automated inspection and maintenance.

Dataset:
    - Source: InfraredSolarModules dataset
    - Total Images: ~20,000 thermal images
    - Classes: 12 fault types
    - Split: 80% train, 10% val, 10% test
"""

from ultralytics import YOLO
import torch
from datetime import datetime
from pathlib import Path

def setup_directories():
    """Create necessary directories for training outputs"""
    dirs = ['runs', 'checkpoints', 'logs', 'results']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("✅ Directories created")

def check_gpu():
    """Check GPU availability and print info"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU Available: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.2f} GB")
        return True
    else:
        print("⚠️  No GPU detected, training on CPU (will be slow)")
        return False

def train_model():
    """Main training function for YOLOv8 classification model"""
    
    # Setup
    setup_directories()
    has_gpu = check_gpu()
    
    # Count images in dataset
    import os
    train_path = Path('data/images/train')
    classes = [d.name for d in train_path.iterdir() if d.is_dir()]
    
    print(f"\n📋 Dataset Configuration:")
    print(f"   Path: data/images")
    print(f"   Classes: {len(classes)}")
    for cls in sorted(classes):
        count = len(list((train_path / cls).glob('*.jpg')))
        print(f"      {cls}: {count} images")
    
    # Initialize YOLOv8 classification model
    model = YOLO('yolov8n-cls.pt')  # Pretrained on ImageNet
    
    print(f"\n🚀 Starting Training...")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Training configuration
    results = model.train(
        data='data/images',
        epochs=100,
        imgsz=224,
        batch=32,
        device=0 if has_gpu else 'cpu',
        
        # Optimization
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        
        # Training settings
        patience=15,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        
        # Logging
        project='runs/classify',
        name='solar_fault_detection',
        exist_ok=True,
        verbose=True,
        
        # Performance
        workers=4,
        amp=True,  # Automatic Mixed Precision
    )
    
    print("\n✅ Training Complete!")
    print(f"   Best model saved at: runs/classify/solar_fault_detection/weights/best.pt")
    
    return model, results

if __name__ == "__main__":
    print("="*70)
    print("SOLAR PANEL FAULT DETECTION - YOLOv8 TRAINING")
    print("="*70)
    
    model, results = train_model()
    
    print("\n📊 Next Steps:")
    print("   1. Run: python evaluate.py  - to evaluate on test set")
    print("   2. Run: python predict.py   - to make predictions")
    print("   3. Check: runs/classify/solar_fault_detection - for logs")
