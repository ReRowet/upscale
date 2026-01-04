#!/usr/bin/env python3
"""
Batch Image Upscaler using Real-ESRGAN
"""

import os
import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

import torch
from PIL import Image
from realesrgan import RealESRGAN

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

class BatchUpscaler:
    def __init__(
        self,
        model_path: str = "weights/realesr-general-x4v3.pth",
        scale: int = 4,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.scale = scale
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """Load Real-ESRGAN model"""
        logger.info(f"Loading model from {self.model_path}")
        self.model = RealESRGAN(self.device, scale=self.scale)
        self.model.load_weights(self.model_path)
        logger.info("Model loaded successfully")
        
    def upscale_image(self, image_path: Path, output_path: Path) -> bool:
        """Upscale single image"""
        try:
            # Open and convert image
            img = Image.open(image_path).convert("RGB")
            
            # Upscale
            sr = self.model.predict(img)
            
            # Save with same name and format
            sr.save(output_path)
            logger.debug(f"Upscaled: {image_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return False
            
    def process_folder(
        self,
        input_folder: Path,
        output_folder: Path,
        cleanup: bool = False
    ) -> dict:
        """Process all images in a folder recursively"""
        results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'failed_files': []
        }
        
        # Ensure output folder exists
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all image files recursively
        image_files = []
        for ext in SUPPORTED_EXTENSIONS:
            image_files.extend(input_folder.rglob(f"*{ext}"))
            image_files.extend(input_folder.rglob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"No images found in {input_folder}")
            return results
            
        results['total'] = len(image_files)
        logger.info(f"Found {len(image_files)} images in {input_folder}")
        
        # Process with progress bar
        for image_path in tqdm(image_files, desc=f"Processing {input_folder.name}"):
            # Calculate relative path for output
            rel_path = image_path.relative_to(input_folder)
            output_path = output_folder / rel_path
            
            # Create parent directories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process image
            if self.upscale_image(image_path, output_path):
                results['success'] += 1
                
                # Delete original if cleanup is enabled
                if cleanup:
                    image_path.unlink()
                    # Remove empty directories
                    try:
                        image_path.parent.rmdir()
                    except OSError:
                        pass
            else:
                results['failed'] += 1
                results['failed_files'].append(str(image_path))
                
        return results
        
    def process_batches(
        self,
        input_root: Path,
        output_root: Path,
        specific_batch: Optional[str] = None,
        cleanup: bool = False
    ) -> dict:
        """
        Process images in batch folders
        
        Args:
            input_root: Root input directory
            output_root: Root output directory
            specific_batch: Process only this batch folder (None for all)
            cleanup: Delete original images after processing
        """
        summary = {}
        
        # If specific batch is provided
        if specific_batch:
            batch_folders = [input_root / specific_batch]
            if not batch_folders[0].exists():
                logger.error(f"Batch folder not found: {specific_batch}")
                return summary
        else:
            # Get all subdirectories in input_root
            batch_folders = [d for d in input_root.iterdir() if d.is_dir()]
            
            # Also include images directly in input_root
            if any(input_root.glob(f"*{ext}") for ext in SUPPORTED_EXTENSIONS):
                batch_folders.append(input_root)
        
        # Load model once
        self.load_model()
        
        # Process each batch
        for batch_folder in batch_folders:
            logger.info(f"Processing batch: {batch_folder.name}")
            
            # Create corresponding output folder
            output_folder = output_root / batch_folder.relative_to(input_root)
            
            # Process the batch
            results = self.process_folder(batch_folder, output_folder, cleanup)
            summary[batch_folder.name] = results
            
            # Log results
            logger.info(
                f"Batch {batch_folder.name}: "
                f"{results['success']}/{results['total']} images upscaled "
                f"({results['failed']} failed)"
            )
            
        return summary

def main():
    parser = argparse.ArgumentParser(description="Batch Image Upscaler")
    parser.add_argument("--input", "-i", default="images",
                       help="Input directory (default: images)")
    parser.add_argument("--output", "-o", default="upscaled",
                       help="Output directory (default: upscaled)")
    parser.add_argument("--batch", "-b", default="",
                       help="Process specific batch folder (default: all)")
    parser.add_argument("--scale", "-s", type=int, default=4,
                       choices=[2, 4], help="Upscale factor (default: 4)")
    parser.add_argument("--model", "-m", default="weights/realesr-general-x4v3.pth",
                       help="Model weights path")
    parser.add_argument("--cleanup", "-c", action="store_true",
                       help="Delete original images after processing")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize upscaler
    upscaler = BatchUpscaler(
        model_path=args.model,
        scale=args.scale,
        device="cpu"
    )
    
    # Convert to Path objects
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return
    
    # Process batches
    batch_name = args.batch if args.batch else None
    summary = upscaler.process_batches(
        input_path,
        output_path,
        specific_batch=batch_name,
        cleanup=args.cleanup
    )
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*50)
    
    total_images = 0
    total_success = 0
    total_failed = 0
    
    for batch_name, results in summary.items():
        logger.info(f"\nBatch: {batch_name}")
        logger.info(f"  Total: {results['total']}")
        logger.info(f"  Success: {results['success']}")
        logger.info(f"  Failed: {results['failed']}")
        
        total_images += results['total']
        total_success += results['success']
        total_failed += results['failed']
        
        if results['failed_files']:
            logger.info(f"  Failed files:")
            for file in results['failed_files']:
                logger.info(f"    - {file}")
    
    logger.info("\n" + "="*50)
    logger.info(f"TOTAL: {total_success}/{total_images} images upscaled")
    logger.info(f"FAILED: {total_failed} images")
    logger.info("="*50)

if __name__ == "__main__":
    main()