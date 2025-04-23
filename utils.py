import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, train_dir, val_dir, test_size=0.2, random_state=42):
    """
    Split image dataset into train/validation sets while maintaining class structure
    
    Args:
        source_dir (str): Path to directory containing 'yes' and 'no' subdirectories
        train_dir (str): Path to output training directory
        val_dir (str): Path to output validation directory
        test_size (float): Proportion of data for validation (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
    """
    # Create output directories
    os.makedirs(os.path.join(train_dir, 'yes'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'no'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'yes'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'no'), exist_ok=True)

    for class_name in ['yes', 'no']:
        # Get list of image files
        class_dir = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split into train/val
        train_files, val_files = train_test_split(
            images, 
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        # Copy files to train directory
        for f in train_files:
            src = os.path.join(class_dir, f)
            dst = os.path.join(train_dir, class_name, f)
            shutil.copyfile(src, dst)
        
        # Copy files to validation directory
        for f in val_files:
            src = os.path.join(class_dir, f)
            dst = os.path.join(val_dir, class_name, f)
            shutil.copyfile(src, dst)
            
        print(f'Class {class_name}:')
        print(f'  Training samples: {len(train_files)}')
        print(f'  Validation samples: {len(val_files)}')

