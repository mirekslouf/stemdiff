# STEM Diffraction Pattern Generator 🔬

A modern, user-friendly tool for generating physically accurate STEM (Scanning Transmission Electron Microscopy) diffraction patterns. Perfect for creating training datasets for machine learning segmentation models.

## ✨ Features

- **Clean Architecture**: Modular design with separated physics, materials database, and generation logic
- **Beautiful GUI**: Modern web interface with real-time preview
- **Batch Generation**: Create large datasets with automatic organization
- **Physical Accuracy**: Proper crystallographic calculations, structure factors, and systematic absences
- **Segmentation Masks**: Automatically generate ground truth masks for ML training
- **Easy to Use**: No more cryptic parameters—just select material and go!

## 🎯 What's New?

This is a complete rewrite of the original monolithic code with:

- ✅ **Crystal Database**: Materials defined in clean JSON format
- ✅ **Simplified API**: Generate patterns in 3 lines of code
- ✅ **Modern GUI**: Beautiful, intuitive interface
- ✅ **Better Organization**: Separate files for physics, generation, and UI
- ✅ **Batch Processing**: Generate 1000s of patterns effortlessly
- ✅ **Real-time Preview**: See results immediately

## 📦 Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Required packages
pip install numpy scipy scikit-image opencv-python pillow flask flask-cors
```

### Quick Start

```bash
# 1. Clone or download all files to a directory
# 2. Navigate to the directory
cd stem-diffraction-generator

# 3. Start the server
python server.py

# 4. Open your browser
# Go to: http://localhost:5000
```

That's it! The GUI will open automatically.

## 🚀 Usage

### Option 1: GUI (Recommended)

1. **Start the server**: Run `python server.py`
2. **Open browser**: Navigate to `http://localhost:5000`
3. **Single Pattern Mode**:
   - Select material
   - Set zone axis (e.g., [1,1,1])
   - Adjust voltage (100-300 kV)
   - Click "Generate Pattern"
   - Download pattern and mask

4. **Batch Dataset Mode**:
   - Select multiple materials
   - Set patterns per material
   - Configure voltage range
   - Click "Generate Dataset"
   - Files saved to `dataset/` folder

### Option 2: Python API

```python
from generator import DiffractionGenerator

# Initialize
gen = DiffractionGenerator()

# Generate single pattern
pattern, mask = gen.generate(
    material='Al',
    zone_axis=[1, 1, 1],
    voltage=200
)

# Generate complete dataset
stats = gen.generate_dataset(
    materials=['Al', 'Cu', 'Si'],
    num_patterns_per_material=100,
    output_dir='my_dataset'
)

print(f"Generated {stats['total_patterns']} patterns!")
```

### Option 3: Command Line

```bash
# Generate single pattern
python -c "from generator import DiffractionGenerator; \
           gen = DiffractionGenerator(); \
           gen.generate('Al', [1,1,1], voltage=200)"

# Generate dataset
python -c "from generator import DiffractionGenerator; \
           gen = DiffractionGenerator(); \
           gen.generate_dataset(['Al', 'Cu'], 100, 'dataset')"
```

## 📁 File Structure

```
stem-diffraction-generator/
├── crystals.json          # Material definitions
├── physics.py             # Core physics calculations
├── generator.py           # Pattern generation engine
├── server.py              # Web server (Flask)
├── diffraction_app.html   # GUI interface
├── README.md              # This file
└── dataset/               # Generated datasets
    ├── images/            # Diffraction patterns
    ├── masks/             # Segmentation masks
    └── metadata.json      # Dataset information
```

## 🔧 Customization

### Adding New Materials

Edit `crystals.json`:

```json
{
  "simple_materials": {
    "MyMaterial": {
      "name": "My Custom Material",
      "structure": "fcc",
      "lattice": {"a": 4.0, "b": 4.0, "c": 4.0, 
                  "alpha": 90, "beta": 90, "gamma": 90},
      "elements": ["My"],
      "atomic_positions": [{"element": "My", "position": [0, 0, 0]}],
      "common_zone_axes": [[0,0,1], [1,1,1]],
      "default_thickness": 40
    }
  }
}
```

### Adjusting Physics Parameters

Modify generator parameters:

```python
pattern, mask = gen.generate(
    material='Al',
    zone_axis=[1, 1, 1],
    voltage=200,
    size=512,              # Higher resolution
    sample_thickness=50,    # Thicker sample
    noise_level=0.03       # More noise
)
```

## 📊 Output Format

### Dataset Structure
```
dataset/
├── images/
│   ├── Al_0000.png
│   ├── Al_0001.png
│   ├── Cu_0000.png
│   └── ...
├── masks/
│   ├── Al_0000.png
│   ├── Al_0001.png
│   ├── Cu_0000.png
│   └── ...
└── metadata.json
```

### Metadata Format
```json
{
  "total_patterns": 300,
  "materials": {
    "Al": 100,
    "Cu": 100,
    "Si": 100
  },
  "parameters": {
    "size": 256,
    "voltage_range": [100, 300],
    "mask_type": "binary"
  }
}
```

## 🎓 Physical Background

This generator uses proper electron diffraction physics:

- **Wavelength Calculation**: Relativistic correction for electron wavelength
- **Reciprocal Lattice**: Accurate calculation of reciprocal space vectors
- **Structure Factors**: Proper atomic scattering factors
- **Systematic Absences**: Crystal structure-specific reflection rules
- **Kinematical Approximation**: Intensity calculations with temperature factors
- **Realistic Noise**: Poisson noise, detector effects, and background

## 🧪 Supported Materials

### Simple Metals
- **FCC**: Al, Cu, Ni, Au
- **BCC**: Fe
- **Diamond**: Si, Ge

### Complex Materials
- **Fe3O4**: Magnetite (Spinel)
- **GaAs**: Gallium Arsenide (Zincblende)
- **SrTiO3**: Strontium Titanate (Perovskite)
- **TiO2**: Rutile Titanium Dioxide
- **ZnO**: Zinc Oxide (Wurtzite)

## 🤝 Contributing

Want to add more materials or features?

1. Add material to `crystals.json`
2. Test with the generator
3. Submit your additions!

### GUI won't load
- Check that server.py is running
- Try a different port: `python server.py --port 8000`
- Clear browser cache



