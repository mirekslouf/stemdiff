#!/usr/bin/env python3
"""
Setup Verification and Helper Script for Diffraction Processing GUI
Checks dependencies and provides helpful error messages
"""

import sys
import subprocess
from pathlib import Path

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)

def check_python_version():
    """Check Python version"""
    print_section("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ ERROR: Python 3.7 or higher is required")
        return False
    else:
        print("✓ Python version is compatible")
        return True

def check_module(module_name, import_name=None):
    """Check if a module is installed"""
    if import_name is None:
        import_name = module_name
    
    try:
        __import__(import_name)
        print(f"✓ {module_name} is installed")
        return True
    except ImportError:
        print(f"❌ {module_name} is NOT installed")
        return False

def check_required_modules():
    """Check all required modules"""
    print_section("Checking Required Python Packages")
    
    required = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'PyQt5': 'PyQt5',
        'scikit-image': 'skimage',
        'opencv-python': 'cv2',
        'scikit-learn': 'sklearn',
        'tqdm': 'tqdm',
        'numba': 'numba'
    }
    
    all_good = True
    missing = []
    
    for package, import_name in required.items():
        if not check_module(package, import_name):
            all_good = False
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
    
    return all_good

def check_optional_modules():
    """Check optional modules"""
    print_section("Checking Optional Python Packages")
    
    optional = {
        'torch': 'torch',
        'pytorch-lightning': 'pytorch_lightning',
    }
    
    for package, import_name in optional.items():
        check_module(package, import_name)

def check_custom_modules():
    """Check custom/local modules"""
    print_section("Checking Custom Modules")
    
    required_files = [
        'Deconv_class.py',
        'utilities.py',
        'PSF_fit.py',
        'psf_function.py',
        'dbase.py',
        'bcorr.py'
    ]
    
    # Check for summ.py or reconstruct.py
    has_summ = Path('summ.py').exists()
    has_reconstruct = Path('reconstruct.py').exists()
    
    all_good = True
    missing = []
    
    for filename in required_files:
        if Path(filename).exists():
            print(f"✓ {filename} found")
        else:
            print(f"❌ {filename} NOT found")
            all_good = False
            missing.append(filename)
    
    if has_summ:
        print(f"✓ summ.py found (reconstruction module)")
    elif has_reconstruct:
        print(f"✓ reconstruct.py found (reconstruction module)")
    else:
        print(f"❌ Neither summ.py nor reconstruct.py found")
        all_good = False
        missing.append("summ.py or reconstruct.py")
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        print("Please copy these files to the current directory")
    
    return all_good

def check_stemdiff():
    """Check if stemdiff library is available"""
    print_section("Checking stemdiff Library")
    
    try:
        import stemdiff
        print("✓ stemdiff library is installed")
        print(f"  Location: {stemdiff.__file__}")
        return True
    except ImportError:
        print("❌ stemdiff library is NOT installed")
        print("  This is a required library for processing .dat files")
        print("  Please install it according to your local setup")
        return False

def check_ediff():
    """Check if ediff library is available"""
    print_section("Checking ediff Library")
    
    try:
        import ediff
        print("✓ ediff library is installed")
        print(f"  Location: {ediff.__file__}")
        return True
    except ImportError:
        print("⚠️  ediff library is NOT installed")
        print("  This is needed for Step 3 (profile comparison)")
        print("  You can still use Steps 1-2 without it")
        return False

def check_bground():
    """Check if bground library is available"""
    print_section("Checking bground Library")
    
    try:
        import bground
        print("✓ bground library is installed")
        print(f"  Location: {bground.__file__}")
        return True
    except ImportError:
        print("⚠️  bground library is NOT installed")
        print("  This is needed for interactive background correction")
        print("  You can skip background correction step without it")
        return False

def provide_summary(results):
    """Provide summary and next steps"""
    print_section("Summary and Next Steps")
    
    all_required = all([
        results['python'],
        results['packages'],
        results['custom'],
        results['stemdiff']
    ])
    
    if all_required:
        print("✅ All required components are installed!")
        print("\nYou can now run the GUI with:")
        print("  python diffraction_gui.py")
    else:
        print("⚠️  Some required components are missing")
        print("\nPlease address the issues above before running the GUI")
    
    if not results['ediff']:
        print("\nNote: ediff is optional but recommended for full functionality")
    
    if not results['bground']:
        print("Note: bground is optional but recommended for background correction")

def main():
    """Main verification function"""
    print("=" * 60)
    print("  Diffraction Processing GUI - Setup Verification")
    print("=" * 60)
    
    results = {
        'python': check_python_version(),
        'packages': check_required_modules(),
        'custom': check_custom_modules(),
        'stemdiff': check_stemdiff(),
        'ediff': check_ediff(),
        'bground': check_bground()
    }
    
    check_optional_modules()
    provide_summary(results)
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()
