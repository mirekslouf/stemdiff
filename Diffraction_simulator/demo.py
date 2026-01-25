"""
Demo script to test the STEM Diffraction Pattern Generator
Run this to verify everything is working correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from generator import DiffractionGenerator


def demo_single_pattern():
    """Demo: Generate and display a single pattern"""
    print("\n🔬 Demo 1: Single Pattern Generation")
    print("=" * 60)
    
    gen = DiffractionGenerator()
    
    # Generate pattern for Aluminum [111]
    print("Generating Al [1,1,1] at 200 kV...")
    pattern, mask = gen.generate(
        material='Al',
        zone_axis=[1, 1, 1],
        voltage=200,
        size=256,
        generate_mask=True
    )
    
    # Display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(pattern, cmap='viridis')
    ax1.set_title('Diffraction Pattern', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Segmentation Mask', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.suptitle('Al [1,1,1] @ 200 kV', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_dir = Path('demo_output')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'demo_single.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: demo_output/demo_single.png")
    
    plt.show()
    plt.close()


def demo_multiple_materials():
    """Demo: Generate patterns for different materials"""
    print("\n🔬 Demo 2: Multiple Materials")
    print("=" * 60)
    
    gen = DiffractionGenerator()
    
    materials = ['Al', 'Cu', 'Si', 'Fe']
    zone_axis = [1, 1, 1]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, material in enumerate(materials):
        print(f"Generating {material} [1,1,1]...")
        
        pattern, mask = gen.generate(
            material=material,
            zone_axis=zone_axis,
            voltage=200,
            size=256,
            generate_mask=True
        )
        
        # Plot pattern
        axes[0, i].imshow(pattern, cmap='viridis')
        axes[0, i].set_title(f'{material} Pattern', fontweight='bold')
        axes[0, i].axis('off')
        
        # Plot mask
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'{material} Mask', fontweight='bold')
        axes[1, i].axis('off')
    
    plt.suptitle('Comparison of Different Materials [1,1,1] @ 200 kV', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_dir = Path('demo_output')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'demo_materials.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: demo_output/demo_materials.png")
    
    plt.show()
    plt.close()


def demo_different_orientations():
    """Demo: Same material, different zone axes"""
    print("\n🔬 Demo 3: Different Orientations")
    print("=" * 60)
    
    gen = DiffractionGenerator()
    
    zone_axes = [[0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 2]]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, za in enumerate(zone_axes):
        print(f"Generating Al {za}...")
        
        pattern, _ = gen.generate(
            material='Al',
            zone_axis=za,
            voltage=200,
            size=256,
            generate_mask=False
        )
        
        axes[i].imshow(pattern, cmap='viridis')
        axes[i].set_title(f'[{za[0]},{za[1]},{za[2]}]', fontweight='bold', fontsize=12)
        axes[i].axis('off')
    
    plt.suptitle('Aluminum at Different Zone Axes @ 200 kV', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_dir = Path('demo_output')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'demo_orientations.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: demo_output/demo_orientations.png")
    
    plt.show()
    plt.close()


def demo_dataset_generation():
    """Demo: Generate small dataset"""
    print("\n🔬 Demo 4: Dataset Generation")
    print("=" * 60)
    
    gen = DiffractionGenerator()
    
    print("Generating small dataset (3 materials x 10 patterns)...")
    
    stats = gen.generate_dataset(
        materials=['Al', 'Cu', 'Si'],
        num_patterns_per_material=10,
        output_dir='demo_dataset',
        voltage_range=(150, 250),
        size=256
    )
    
    print("\n✅ Dataset generated successfully!")
    print(f"   Total patterns: {stats['total_patterns']}")
    print(f"   Location: demo_dataset/")
    print(f"   - images/ : {stats['total_patterns']} diffraction patterns")
    print(f"   - masks/  : {stats['total_patterns']} segmentation masks")
    print(f"   - metadata.json : dataset information")
    
    # Show sample of generated images
    from PIL import Image
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    for i in range(3):
        # Load image and mask
        img_path = Path('demo_dataset') / 'images' / f"{list(stats['materials'].keys())[i]}_0000.png"
        mask_path = Path('demo_dataset') / 'masks' / f"{list(stats['materials'].keys())[i]}_0000.png"
        
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        axes[0, i].imshow(img, cmap='viridis')
        axes[0, i].set_title(f"{list(stats['materials'].keys())[i]} Pattern", fontweight='bold')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f"{list(stats['materials'].keys())[i]} Mask", fontweight='bold')
        axes[1, i].axis('off')
    
    plt.suptitle('Sample from Generated Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('demo_output/demo_dataset_sample.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: demo_output/demo_dataset_sample.png")
    
    plt.show()
    plt.close()


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("🚀 STEM Diffraction Pattern Generator - Demo")
    print("=" * 60)
    
    try:
        # Run demos
        demo_single_pattern()
        demo_multiple_materials()
        demo_different_orientations()
        demo_dataset_generation()
        
        print("\n" + "=" * 60)
        print("✨ All demos completed successfully!")
        print("=" * 60)
        print("\n📁 Check the following directories:")
        print("   - demo_output/  : visualization plots")
        print("   - demo_dataset/ : sample dataset")
        print("\n🎉 The system is working correctly!")
        print("\nNext steps:")
        print("   1. Run 'python server.py' to start the GUI")
        print("   2. Or use 'python cli.py' for command-line usage")
        print("   3. Read README.md for more information\n")
        
    except Exception as e:
        print(f"\n❌ Error running demos: {e}")
        print("\nPlease check:")
        print("   1. All required packages are installed")
        print("   2. crystals.json is in the current directory")
        print("   3. All Python files are in the same directory\n")


if __name__ == '__main__':
    main()
