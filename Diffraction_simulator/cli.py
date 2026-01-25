#!/usr/bin/env python3
"""
Command-line interface for STEM Diffraction Pattern Generator
Easy-to-use script for generating patterns without GUI
"""

import argparse
import sys
from pathlib import Path

from generator import DiffractionGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Generate STEM diffraction patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single pattern
  python cli.py single Al -z 1 1 1 -v 200

  # Generate dataset
  python cli.py batch -m Al Cu Si -n 100

  # List available materials
  python cli.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List materials command
    list_parser = subparsers.add_parser('list', help='List available materials')
    
    # Single pattern command
    single_parser = subparsers.add_parser('single', help='Generate single pattern')
    single_parser.add_argument('material', help='Material name (e.g., Al, Cu, Si)')
    single_parser.add_argument('-z', '--zone-axis', nargs=3, type=int, default=[1, 1, 1],
                              metavar=('h', 'k', 'l'), help='Zone axis [h k l]')
    single_parser.add_argument('-v', '--voltage', type=float, default=200,
                              help='Accelerating voltage in kV (default: 200)')
    single_parser.add_argument('-s', '--size', type=int, default=256,
                              help='Image size in pixels (default: 256)')
    single_parser.add_argument('-n', '--noise', type=float, default=0.02,
                              help='Noise level 0-1 (default: 0.02)')
    single_parser.add_argument('-o', '--output', default='output',
                              help='Output directory (default: output)')
    single_parser.add_argument('--no-mask', action='store_true',
                              help='Do not generate mask')
    
    # Batch dataset command
    batch_parser = subparsers.add_parser('batch', help='Generate batch dataset')
    batch_parser.add_argument('-m', '--materials', nargs='+', required=True,
                             help='Materials to include (e.g., Al Cu Si)')
    batch_parser.add_argument('-n', '--num-patterns', type=int, default=100,
                             help='Patterns per material (default: 100)')
    batch_parser.add_argument('-o', '--output', default='dataset',
                             help='Output directory (default: dataset)')
    batch_parser.add_argument('--voltage-min', type=float, default=100,
                             help='Minimum voltage in kV (default: 100)')
    batch_parser.add_argument('--voltage-max', type=float, default=300,
                             help='Maximum voltage in kV (default: 300)')
    batch_parser.add_argument('-s', '--size', type=int, default=256,
                             help='Image size in pixels (default: 256)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize generator
    try:
        gen = DiffractionGenerator()
    except Exception as e:
        print(f"❌ Error initializing generator: {e}")
        print("Make sure crystals.json is in the current directory")
        return
    
    # Execute command
    if args.command == 'list':
        print("\n📋 Available Materials:")
        print("=" * 60)
        
        materials = gen.get_available_materials()
        for mat in materials:
            info = gen.get_material_info(mat)
            print(f"\n  {mat:10s} - {info['name']}")
            print(f"             Structure: {info['structure']}")
            print(f"             Common axes: {info['common_zone_axes'][:3]}")
        
        print("\n" + "=" * 60)
        print(f"Total: {len(materials)} materials\n")
    
    elif args.command == 'single':
        print("\n🔬 Generating Single Pattern")
        print("=" * 60)
        print(f"Material:    {args.material}")
        print(f"Zone Axis:   [{args.zone_axis[0]}, {args.zone_axis[1]}, {args.zone_axis[2]}]")
        print(f"Voltage:     {args.voltage} kV")
        print(f"Size:        {args.size}x{args.size} px")
        print(f"Noise:       {args.noise}")
        print(f"Output:      {args.output}/")
        print("=" * 60)
        
        try:
            # Generate pattern
            pattern, mask = gen.generate(
                material=args.material,
                zone_axis=args.zone_axis,
                voltage=args.voltage,
                size=args.size,
                noise_level=args.noise,
                generate_mask=not args.no_mask
            )
            
            # Save outputs
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            from PIL import Image
            
            # Save pattern
            pattern_path = output_dir / f"{args.material}_pattern.png"
            Image.fromarray((pattern * 255).astype('uint8')).save(pattern_path)
            print(f"\n✅ Saved pattern:  {pattern_path}")
            
            # Save mask
            if not args.no_mask and mask is not None:
                mask_path = output_dir / f"{args.material}_mask.png"
                Image.fromarray((mask * 255).astype('uint8')).save(mask_path)
                print(f"✅ Saved mask:     {mask_path}")
            
            print("\n✨ Done!\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            return
    
    elif args.command == 'batch':
        print("\n🔬 Generating Batch Dataset")
        print("=" * 60)
        print(f"Materials:       {', '.join(args.materials)}")
        print(f"Patterns/mat:    {args.num_patterns}")
        print(f"Voltage range:   {args.voltage_min}-{args.voltage_max} kV")
        print(f"Size:            {args.size}x{args.size} px")
        print(f"Output:          {args.output}/")
        print(f"Total patterns:  {len(args.materials) * args.num_patterns}")
        print("=" * 60)
        
        try:
            # Generate dataset
            print("\n⏳ Generating... (this may take a while)\n")
            
            stats = gen.generate_dataset(
                materials=args.materials,
                num_patterns_per_material=args.num_patterns,
                output_dir=args.output,
                voltage_range=(args.voltage_min, args.voltage_max),
                size=args.size
            )
            
            print("\n✅ Dataset Generation Complete!")
            print("=" * 60)
            print(f"Total patterns:  {stats['total_patterns']}")
            print("\nBreakdown by material:")
            for mat, count in stats['materials'].items():
                print(f"  {mat:10s} - {count} patterns")
            
            print(f"\n📁 Output location: {args.output}/")
            print("   ├── images/     (diffraction patterns)")
            print("   ├── masks/      (segmentation masks)")
            print("   └── metadata.json")
            print("\n✨ Done!\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            return


if __name__ == '__main__':
    main()
