import torch
import os
import traceback

from models.models import MobileViTCo
from data.dataset import create_data_loaders
from trainer import train_model, evaluate_model
from utils.utils import analyze_model

def main_pipeline():
    
    # config 
    DATA_DIR = "/kaggle/input/haze-images/dataset_dehazed/dataset_dehazed"  
    BATCH_SIZE = 32
    NUM_EPOCHS = 150 
    LEARNING_RATE = 0.001
    MODEL_VARIANT = 'XS' 
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    NUM_WORKERS = None 
    PATIENCE_EARLY_STOPPING = 15 
    MODEL_SAVE_PATH = 'best_color_model.pth'
    CONF_MATRIX_SAVE_PATH = 'confusion_matrix.png'
    TRAINING_CURVES_SAVE_PATH = 'training_curves.png'

    print("=== Color Classification Training Pipeline ===")
    print(f"Dataset directory: {DATA_DIR}")
    print(f"Model variant: {MODEL_VARIANT}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print("=" * 50)
    
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found or is empty!")
        print("Please ensure the dataset is correctly placed and structured.")
        print("Expected structure: DATA_DIR/class1/image1.jpg, DATA_DIR/class2/imageA.png etc.")

        return
    
    try:
        print("Creating data loaders...")
        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            DATA_DIR, 
            batch_size=BATCH_SIZE,
            val_split=VAL_SPLIT,
            test_split=TEST_SPLIT,
            num_workers=NUM_WORKERS
        )
        
        print(f"Classes found: {class_names} (Number of classes: {len(class_names)})")
        
        print(f"\nCreating {MODEL_VARIANT} model...")
        num_classes = len(class_names)
        if num_classes == 0:
            raise ValueError("No classes found. Check dataset structure and `ColorDataset` class.")
            
        model = MobileViTCo(num_classes=num_classes, variant=MODEL_VARIANT)
        
        analyze_model(model)
        
        print(f"\nStarting training for up to {NUM_EPOCHS} epochs...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        trained_model = train_model(
            model, train_loader, val_loader,
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            use_mixup=True,         
            label_smoothing=0.1,    
            warmup_epochs=5,        
            patience=PATIENCE_EARLY_STOPPING,
            device=device,
            model_save_path=MODEL_SAVE_PATH
        )
        
        # evaluate
        print("\nEvaluating on test set...")
        test_accuracy, _, _, _ = evaluate_model(
            trained_model, 
            test_loader, 
            class_names, 
            device,
            conf_matrix_path=CONF_MATRIX_SAVE_PATH
        )
        
        print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")
        print(f"Training completed! Check '{TRAINING_CURVES_SAVE_PATH}' and '{CONF_MATRIX_SAVE_PATH}' for results.")
        print(f"Best model saved as '{MODEL_SAVE_PATH}'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main_pipeline()