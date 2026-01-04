#!/usr/bin/env python3
"""
Real-ESRGAN Image Upscaler
Usage: python upscale.py --input images --output upscaled --model models/realesr-general-wdn-x4v3.pth
"""

import os
import sys
import argparse
import glob
import traceback
from pathlib import Path

# Import Real-ESRGAN
try:
    import cv2
    import torch
    import numpy as np
    from PIL import Image
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    from realesrgan import RealESRGANer
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install: pip install basicsr realesrgan opencv-python-headless Pillow")
    sys.exit(1)

def determine_model_type(model_path):
    """Determine model architecture from filename"""
    model_name = Path(model_path).name.lower()
    
    if 'realesr-general-x4v3' in model_name or 'realesr-general-wdn-x4v3' in model_name:
        return 'SRVGGNetCompact', 4
    elif 'realesr-animevideov3' in model_name:
        return 'SRVGGNetCompact', 4
    else:
        # Default to SRVGGNetCompact
        return 'SRVGGNetCompact', 4

def upscale_image(img_path, output_path, upsampler, scale=4):
    """Upscale single image"""
    try:
        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  ✗ Failed to read image")
            return False
        
        # Upscale
        output, _ = upsampler.enhance(img, outscale=scale)
        
        # Save
        cv2.imwrite(str(output_path), output)
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Real-ESRGAN Image Upscaler')
    parser.add_argument('--input', '-i', default='images', help='Input directory')
    parser.add_argument('--output', '-o', default='upscaled', help='Output directory')
    parser.add_argument('--model', '-m', default='models/realesr-general-wdn-x4v3.pth', help='Model path')
    parser.add_argument('--scale', '-s', type=int, default=4, choices=[2, 4], help='Upscale factor')
    parser.add_argument('--tile', type=int, default=0, help='Tile size (0 for no tiling)')
    parser.add_argument('--ext', default='.png,.jpg,.jpeg,.bmp,.webp', help='Image extensions')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Real-ESRGAN Image Upscaler")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"Scale: {args.scale}x")
    print("=" * 60)
    
    # Check paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)
    
    if not input_path.exists():
        print(f"✗ Error: Input directory '{input_path}' not found!")
        sys.exit(1)
    
    if not model_path.exists():
        print(f"✗ Error: Model file '{model_path}' not found!")
        print("  Make sure the model is in the models/ directory")
        sys.exit(1)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine model type
    model_type, model_scale = determine_model_type(model_path)
    print(f"Model type: {model_type}, Native scale: {model_scale}x")
    
    # Initialize model
    print("\nLoading Real-ESRGAN model...")
    try:
        if model_type == 'SRVGGNetCompact':
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=model_scale,
                act_type='prelu'
            )
        
        upsampler = RealESRGANer(
            scale=model_scale,
            model_path=str(model_path),
            model=model,
            tile=args.tile,
            tile_pad=10,
            pre_pad=0,
            half=False,  # CPU mode
            device=torch.device('cpu')
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Find images
    extensions = args.ext.split(',')
    image_files = []
    
    for ext in extensions:
        ext = ext.strip()
        # Search recursively
        pattern = str(input_path / '**' / f'*{ext}')
        image_files.extend(glob.glob(pattern, recursive=True))
        # Case insensitive
        pattern = str(input_path / '**' / f'*{ext.upper()}')
        image_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"✗ No images found in '{input_path}'")
        print(f"  Supported extensions: {args.ext}")
        sys.exit(1)
    
    print(f"\nFound {len(image_files)} images")
    
    # Process images
    success_count = 0
    
    for i, img_file in enumerate(image_files, 1):
        img_path = Path(img_file)
        # Preserve directory structure
        rel_path = img_path.relative_to(input_path)
        output_file = output_path / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[{i}/{len(image_files)}] {rel_path}")
        
        if upscale_image(img_path, output_file, upsampler, args.scale):
            success_count += 1
            print(f"  ✓ Saved to {rel_path}")
        else:
            print(f"  ✗ Failed")
    
    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total images: {len(image_files)}")
    print(f"Successfully upscaled: {success_count}")
    print(f"Failed: {len(image_files) - success_count}")
    print(f"Output directory: {output_path}")
    print("=" * 60)
    
    if success_count == 0:
        print("\n✗ No images were processed successfully!")
        sys.exit(1)
    
    print("\n✅ Upscaling completed successfully!")
    
    # Create manifest file
    manifest_file = output_path / "manifest.txt"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        f.write(f"Real-ESRGAN Upscaling Results\n")
        f.write(f"{'='*40}\n")
        f.write(f"Model: {model_path.name}\n")
        f.write(f"Scale: {args.scale}x\n")
        f.write(f"Input directory: {input_path}\n")
        f.write(f"Output directory: {output_path}\n")
        f.write(f"Total images: {len(image_files)}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Failed: {len(image_files) - success_count}\n\n")
        f.write("Processed files:\n")
        for img_file in image_files:
            img_path = Path(img_file)
            rel_path = img_path.relative_to(input_path)
            output_file = output_path / rel_path
            if output_file.exists():
                f.write(f"[✓] {rel_path}\n")
            else:
                f.write(f"[✗] {rel_path}\n")
    
    print(f"Manifest created: {manifest_file}")

if __name__ == "__main__":
    main()