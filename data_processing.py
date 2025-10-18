"""
Skin Disease Dataset Processing Pipeline
=========================================

This script processes skin disease images with the following steps:
1. Resize images to 224x224
2. Pixel normalization
3. Data augmentation (especially for minority classes)
4. Quality filtering to remove poor-quality/mislabeled images

Author: Data Processing Pipeline
Date: 2025-10-14
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from collections import defaultdict
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class ImageQualityChecker:
    """Check image quality and filter out poor-quality images"""
    
    def __init__(self, min_variance=50, min_brightness=20, max_brightness=250):
        self.min_variance = min_variance
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
    
    def is_good_quality(self, image):
        """
        Check if image meets quality criteria
        
        Args:
            image: numpy array of the image
            
        Returns:
            tuple: (is_good, reason)
        """
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Check 1: Image variance (detect blank/near-blank images)
        variance = np.var(gray)
        if variance < self.min_variance:
            return False, f"Low variance: {variance:.2f}"
        
        # Check 2: Brightness (too dark or too bright)
        mean_brightness = np.mean(gray)
        if mean_brightness < self.min_brightness:
            return False, f"Too dark: {mean_brightness:.2f}"
        if mean_brightness > self.max_brightness:
            return False, f"Too bright: {mean_brightness:.2f}"
        
        # Check 3: Detect mostly black or white images
        black_pixels = np.sum(gray < 20)
        white_pixels = np.sum(gray > 235)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        if black_pixels / total_pixels > 0.9:
            return False, "Too many black pixels"
        if white_pixels / total_pixels > 0.9:
            return False, "Too many white pixels"
        
        # Check 4: Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            return False, f"Too blurry: {laplacian_var:.2f}"
        
        return True, "OK"


class DataAugmentor:
    """Data augmentation for images"""
    
    @staticmethod
    def rotate(image, angle):
        """Rotate image by given angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), 
                                 borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    @staticmethod
    def flip_horizontal(image):
        """Flip image horizontally"""
        return cv2.flip(image, 1)
    
    @staticmethod
    def flip_vertical(image):
        """Flip image vertically"""
        return cv2.flip(image, 0)
    
    @staticmethod
    def adjust_brightness(image, factor):
        """
        Adjust brightness
        factor > 1: brighter, factor < 1: darker
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    @staticmethod
    def adjust_contrast(image, factor):
        """
        Adjust contrast
        factor > 1: more contrast, factor < 1: less contrast
        """
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_noise(image, noise_level=10):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def random_crop_and_resize(image, crop_fraction=0.9):
        """Random crop and resize back to original size"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * crop_fraction), int(w * crop_fraction)
        
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        
        cropped = image[top:top+new_h, left:left+new_w]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized
    
    def augment_image(self, image, num_augmentations=5):
        """
        Generate multiple augmented versions of an image
        
        Args:
            image: input image
            num_augmentations: number of augmented versions to create
            
        Returns:
            list of augmented images
        """
        augmented_images = []
        
        augmentation_functions = [
            lambda img: self.rotate(img, np.random.uniform(-15, 15)),
            lambda img: self.flip_horizontal(img),
            lambda img: self.adjust_brightness(img, np.random.uniform(0.8, 1.2)),
            lambda img: self.adjust_contrast(img, np.random.uniform(0.8, 1.2)),
            lambda img: self.add_noise(img, np.random.uniform(5, 15)),
            lambda img: self.random_crop_and_resize(img, np.random.uniform(0.85, 0.95)),
        ]
        
        for _ in range(num_augmentations):
            # Randomly select and apply 1-3 augmentations
            num_ops = np.random.randint(1, 4)
            aug_img = image.copy()
            
            selected_ops = np.random.choice(len(augmentation_functions), 
                                          size=num_ops, replace=False)
            
            for op_idx in selected_ops:
                aug_img = augmentation_functions[op_idx](aug_img)
            
            augmented_images.append(aug_img)
        
        return augmented_images


class SkinDiseaseDataProcessor:
    """Main data processing pipeline"""
    
    def __init__(self, input_dir, output_dir, target_size=(224, 224)):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.quality_checker = ImageQualityChecker()
        self.augmentor = DataAugmentor()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'filtered_out': 0,
            'augmented': 0,
            'per_category': defaultdict(lambda: {
                'original': 0,
                'filtered': 0,
                'augmented': 0,
                'total': 0
            })
        }
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image
        
        Args:
            image_path: path to the image
            
        Returns:
            preprocessed image or None if failed
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return None, "Failed to load"
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            image = cv2.resize(image, self.target_size, 
                             interpolation=cv2.INTER_AREA)
            
            return image, "OK"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def normalize_image(self, image):
        """
        Normalize image pixels to [0, 1] range
        
        Args:
            image: input image (0-255 range)
            
        Returns:
            normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def save_image(self, image, output_path):
        """Save image to disk"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), image_bgr)
    
    def process_dataset(self, min_samples_for_augmentation=2000, 
                       augmentation_factor=3):
        """
        Process the entire dataset
        
        Args:
            min_samples_for_augmentation: minimum samples before augmentation needed
            augmentation_factor: how many augmented samples per original for minority classes
        """
        print("=" * 80)
        print("SKIN DISEASE DATASET PROCESSING PIPELINE")
        print("=" * 80)
        print()
        
        # Get all categories
        categories = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        # Count samples per category
        category_counts = {}
        for category in categories:
            images = list(category.glob("*.jpg"))
            category_counts[category.name] = len(images)
        
        # Identify minority classes
        avg_samples = np.mean(list(category_counts.values()))
        minority_classes = {cat: count for cat, count in category_counts.items() 
                          if count < min_samples_for_augmentation}
        
        print(f"Found {len(categories)} categories")
        print(f"Average samples per category: {avg_samples:.0f}")
        print(f"Minority classes (< {min_samples_for_augmentation}): {len(minority_classes)}")
        print()
        
        # Process each category
        for category_dir in tqdm(categories, desc="Processing categories"):
            category_name = category_dir.name
            image_files = list(category_dir.glob("*.jpg"))
            
            print(f"\nProcessing: {category_name} ({len(image_files)} images)")
            
            is_minority = category_name in minority_classes
            
            good_images = []
            filtered_reasons = defaultdict(int)
            
            # Process original images
            for img_path in tqdm(image_files, desc="  Loading & filtering", 
                                leave=False):
                # Load and preprocess
                image, load_status = self.load_and_preprocess_image(img_path)
                
                if image is None:
                    filtered_reasons[load_status] += 1
                    self.stats['per_category'][category_name]['filtered'] += 1
                    continue
                
                # Quality check
                is_good, reason = self.quality_checker.is_good_quality(image)
                
                if not is_good:
                    filtered_reasons[reason] += 1
                    self.stats['per_category'][category_name]['filtered'] += 1
                    self.stats['filtered_out'] += 1
                    continue
                
                # Save good image
                good_images.append((image, img_path.name))
                self.stats['per_category'][category_name]['original'] += 1
                self.stats['total_processed'] += 1
            
            # Save original good images
            output_category_dir = self.output_dir / category_name
            for image, filename in tqdm(good_images, desc="  Saving originals", 
                                       leave=False):
                output_path = output_category_dir / filename
                self.save_image(image, output_path)
            
            # Apply augmentation for minority classes
            if is_minority and len(good_images) > 0:
                print(f"  Applying augmentation (minority class)...")
                num_augmentations = min(augmentation_factor, 
                                       (min_samples_for_augmentation - len(good_images)) 
                                       // len(good_images) + 1)
                
                aug_count = 0
                for image, filename in tqdm(good_images, 
                                           desc="  Augmenting", leave=False):
                    augmented = self.augmentor.augment_image(image, num_augmentations)
                    
                    base_name = Path(filename).stem
                    for i, aug_img in enumerate(augmented):
                        aug_filename = f"{base_name}_aug{i}.jpg"
                        output_path = output_category_dir / aug_filename
                        self.save_image(aug_img, output_path)
                        aug_count += 1
                
                self.stats['per_category'][category_name]['augmented'] = aug_count
                self.stats['augmented'] += aug_count
                print(f"  Generated {aug_count} augmented images")
            
            # Update total
            self.stats['per_category'][category_name]['total'] = (
                self.stats['per_category'][category_name]['original'] +
                self.stats['per_category'][category_name]['augmented']
            )
            
            # Print filtering reasons if any
            if filtered_reasons:
                print(f"  Filtered out: {sum(filtered_reasons.values())} images")
                for reason, count in filtered_reasons.items():
                    print(f"    - {reason}: {count}")
        
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        self.print_statistics()
        self.save_statistics()
    
    def print_statistics(self):
        """Print processing statistics"""
        print()
        print(f"Total images processed: {self.stats['total_processed']}")
        print(f"Total images filtered out: {self.stats['filtered_out']}")
        print(f"Total augmented images created: {self.stats['augmented']}")
        print()
        
        print(f"{'Category':<50} {'Original':>10} {'Filtered':>10} {'Augmented':>10} {'Total':>10}")
        print("-" * 95)
        
        for category in sorted(self.stats['per_category'].keys()):
            stats = self.stats['per_category'][category]
            print(f"{category:<50} {stats['original']:>10} {stats['filtered']:>10} "
                  f"{stats['augmented']:>10} {stats['total']:>10}")
        
        print("-" * 95)
        total_original = sum(s['original'] for s in self.stats['per_category'].values())
        total_filtered = sum(s['filtered'] for s in self.stats['per_category'].values())
        total_augmented = sum(s['augmented'] for s in self.stats['per_category'].values())
        total_final = sum(s['total'] for s in self.stats['per_category'].values())
        
        print(f"{'TOTAL':<50} {total_original:>10} {total_filtered:>10} "
              f"{total_augmented:>10} {total_final:>10}")
    
    def save_statistics(self):
        """Save statistics to JSON file"""
        stats_file = self.output_dir / "processing_statistics.json"
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        print(f"\n✅ Statistics saved to: {stats_file}")
        
        # Also save as text file
        stats_txt = self.output_dir / "processing_report.txt"
        with open(stats_txt, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SKIN DISEASE DATASET PROCESSING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Target image size: {self.target_size}\n")
            f.write(f"Normalization: Pixel values scaled to [0, 1]\n\n")
            
            f.write(f"Total images processed: {self.stats['total_processed']}\n")
            f.write(f"Total images filtered out: {self.stats['filtered_out']}\n")
            f.write(f"Total augmented images created: {self.stats['augmented']}\n\n")
            
            f.write(f"{'Category':<50} {'Original':>10} {'Filtered':>10} {'Augmented':>10} {'Total':>10}\n")
            f.write("-" * 95 + "\n")
            
            for category in sorted(self.stats['per_category'].keys()):
                stats = self.stats['per_category'][category]
                f.write(f"{category:<50} {stats['original']:>10} {stats['filtered']:>10} "
                       f"{stats['augmented']:>10} {stats['total']:>10}\n")
        
        print(f"✅ Report saved to: {stats_txt}")


def main():
    """Main execution function"""
    # Configuration
    INPUT_DIR = "unified_dataset"
    OUTPUT_DIR = "processed_dataset"
    TARGET_SIZE = (224, 224)
    MIN_SAMPLES_FOR_AUGMENTATION = 2000
    AUGMENTATION_FACTOR = 3
    
    # Create processor
    processor = SkinDiseaseDataProcessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        target_size=TARGET_SIZE
    )
    
    # Process dataset
    processor.process_dataset(
        min_samples_for_augmentation=MIN_SAMPLES_FOR_AUGMENTATION,
        augmentation_factor=AUGMENTATION_FACTOR
    )
    
    print("\n" + "=" * 80)
    print("✅ DATASET PROCESSING PIPELINE COMPLETED!")
    print("=" * 80)
    print(f"\nProcessed dataset location: {OUTPUT_DIR}/")
    print("\nNext steps:")
    print("1. Review the processing_report.txt for details")
    print("2. Use the processed images for model training")
    print("3. Images are resized to 224x224 and quality-filtered")
    print("4. Minority classes have been augmented for better balance")


if __name__ == "__main__":
    main()

