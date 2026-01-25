# Diffraction Data Processing GUI

A comprehensive graphical user interface for processing electron diffraction data with multiple reconstruction methods and theoretical profile comparison.

## Features

### Step 1: Data Loading and Database Generation
- Automatic database calculation from .dat files
- PSF (Point Spread Function) estimation
- Quality control plots
- Optimal file selection based on quality metrics

### Step 2: Reconstruction and Processing
Multiple reconstruction methods:
1. **Raw Sum** - Simple summation
2. **Richardson-Lucy** - Deconvolution with PSF
3. **GMM Segmentation** - Gaussian Mixture Model
4. **Row Thresholding** - Polar coordinate segmentation
5. **Peak Finding** - Blob detection (LoG/DoH/MSER/PCBR)
6. **UNet** - Deep learning segmentation

### Step 3: Profile Calculation and Comparison
- Radial profile calculation
- Theoretical PXRD from CIF files
- Interactive background correction
- Automatic calibration and comparison

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place required modules in same directory:
# - Deconv_class.py
# - utilities.py  
# - PSF_fit.py
# - psf_function.py
# - summ.py (or reconstruct.py)
# - dbase.py
# - bcorr.py
```

## Usage

```bash
python diffraction_gui.py
```

### Quick Start Guide

1. **Step 1**: Browse for data folder → Configure parameters → Generate Database
2. **Step 2**: Select method → Configure parameters → Run Processing  
3. **Step 3**: Calculate profiles → Background correction → Generate comparison

## Method Guide

### Richardson-Lucy
- **Best for**: Removing PSF blur
- **Iterations**: 10-50 typical
- **Regularization**: TM for noisy data
- **Lambda**: 0.01-0.1

### Row Thresholding  
- **Best for**: Robust peak detection
- **Start Row**: 50-70 (skip main beam)
- **Threshold Factor**: 2-5 (MAD multiplier)
- **Min Blob Size**: 20-50 pixels

### Peak Finding (DoH)
- **Best for**: Precise localization
- **Detector**: 'doh' recommended
- **Background**: 'opening_downsampled' for speed
- **Sigma Range**: 6-60 pixels

## File Organization

```
results/
├── dbase_all.zip              # Full database
├── dbase_sum.zip              # Selected files
├── psf.npy                    # PSF
├── *_result.png               # Processed diffractogram
├── ed_radial_profile.txt      # 1D profile
├── comparison_plot.png        # Final comparison
└── QC plots...
```

## Performance

Processing times for 200 files:

| Method | Time | Accuracy |
|--------|------|----------|
| GMM | 5-10 min | Medium |
| Row Threshold | 3-7 min | High |
| DoH | 10-20 min | Very High |

## Troubleshooting

**"Module not found"**: Ensure custom modules are in same directory
**"No .dat files"**: Check folder path and file pattern
**Slow processing**: Reduce file count or use faster methods
**Memory errors**: Process in smaller batches

## Requirements

- Python 3.7+
- PyQt5, numpy, scipy, matplotlib
- scikit-image, opencv-python
- stemdiff, ediff libraries
- Optional: PyTorch for UNet


