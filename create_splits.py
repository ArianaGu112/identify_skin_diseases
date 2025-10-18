"""
Create Train/Validation/Test Splits
====================================

This script splits the processed dataset into train, validation, and test sets.
Default split: 70% train, 15% validation, 15% test
"""

import os
import shutil
from pathlib import Path
import random
import json
from collections import defaultdict
from tqdm import tqdm

def create_train_val_test_splits(input_dir, output_dir, 
                                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                  random_seed=42):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        input_dir: Directory containing processed images organized by category
        output_dir: Output directory for split datasets
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
    """
    
    random.seed(random_seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CREATING TRAIN/VALIDATION/TEST SPLITS")
    print("=" * 80)
    print(f"Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%")
    print()
    
    # Statistics
    stats = {
        'total': 0,
        'train': 0,
        'val': 0,
        'test': 0,
        'per_category': defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})
    }
    
    # Get all categories
    categories = [d for d in input_path.iterdir() 
                 if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(categories)} categories")
    print()
    
    # Process each category
    for category_dir in tqdm(categories, desc="Processing categories"):
        category_name = category_dir.name
        
        # Get all images
        image_files = list(category_dir.glob("*.jpg"))
        random.shuffle(image_files)
        
        total_images = len(image_files)
        if total_images == 0:
            continue
        
        # Calculate split indices
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)
        
        # Split files
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # Copy files to respective directories
        for split_name, files in [('train', train_files), 
                                  ('val', val_files), 
                                  ('test', test_files)]:
            
            split_category_dir = output_path / split_name / category_name
            split_category_dir.mkdir(parents=True, exist_ok=True)
            
            for img_file in files:
                shutil.copy2(img_file, split_category_dir / img_file.name)
            
            # Update statistics
            stats[split_name] += len(files)
            stats['per_category'][category_name][split_name] = len(files)
        
        # Update total statistics
        stats['total'] += total_images
        stats['per_category'][category_name]['total'] = total_images
    
    print("\n" + "=" * 80)
    print("SPLIT STATISTICS")
    print("=" * 80)
    print()
    
    # Print overall statistics
    print(f"Total images: {stats['total']:,}")
    print(f"Training set: {stats['train']:,} ({stats['train']/stats['total']*100:.1f}%)")
    print(f"Validation set: {stats['val']:,} ({stats['val']/stats['total']*100:.1f}%)")
    print(f"Test set: {stats['test']:,} ({stats['test']/stats['total']*100:.1f}%)")
    print()
    
    # Print per-category statistics
    print(f"{'Category':<50} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print("-" * 95)
    
    for category in sorted(stats['per_category'].keys()):
        cat_stats = stats['per_category'][category]
        print(f"{category:<50} {cat_stats['train']:>10} {cat_stats['val']:>10} "
              f"{cat_stats['test']:>10} {cat_stats['total']:>10}")
    
    print("-" * 95)
    print(f"{'TOTAL':<50} {stats['train']:>10} {stats['val']:>10} "
          f"{stats['test']:>10} {stats['total']:>10}")
    
    # Save statistics
    stats_file = output_path / "split_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Save report
    report_file = output_path / "split_report.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAIN/VALIDATION/TEST SPLIT REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Split ratios: Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%\n")
        f.write(f"Random seed: {random_seed}\n\n")
        
        f.write(f"Total images: {stats['total']:,}\n")
        f.write(f"Training set: {stats['train']:,} ({stats['train']/stats['total']*100:.1f}%)\n")
        f.write(f"Validation set: {stats['val']:,} ({stats['val']/stats['total']*100:.1f}%)\n")
        f.write(f"Test set: {stats['test']:,} ({stats['test']/stats['total']*100:.1f}%)\n\n")
        
        f.write(f"{'Category':<50} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}\n")
        f.write("-" * 95 + "\n")
        
        for category in sorted(stats['per_category'].keys()):
            cat_stats = stats['per_category'][category]
            f.write(f"{category:<50} {cat_stats['train']:>10} {cat_stats['val']:>10} "
                   f"{cat_stats['test']:>10} {cat_stats['total']:>10}\n")
    
    print(f"\n✅ Statistics saved to: {stats_file}")
    print(f"✅ Report saved to: {report_file}")
    print("\n" + "=" * 80)
    print("✅ SPLIT CREATION COMPLETE!")
    print("=" * 80)
    print(f"\nDataset location: {output_dir}/")
    print("  - train/")
    print("  - val/")
    print("  - test/")


def main():
    """Main execution"""
    INPUT_DIR = "processed_dataset"
    OUTPUT_DIR = "final_dataset"
    
    create_train_val_test_splits(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )


if __name__ == "__main__":
    main()

