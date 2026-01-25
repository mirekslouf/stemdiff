"""
Backend server for STEM Diffraction Pattern Generator
Connects the GUI with the pattern generation engine
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
import json

from generator import DiffractionGenerator

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize generator
generator = DiffractionGenerator()


def numpy_to_base64(array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded image"""
    # Normalize to 0-255
    image_data = (array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    image = Image.fromarray(image_data)
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@app.route('/api/materials', methods=['GET'])
def get_materials():
    """Get list of available materials"""
    materials = generator.get_available_materials()
    
    # Get detailed info for each material
    materials_info = {}
    for mat in materials:
        info = generator.get_material_info(mat)
        materials_info[mat] = {
            'name': info.get('name', mat),
            'structure': info['structure'],
            'common_zone_axes': info['common_zone_axes']
        }
    
    return jsonify(materials_info)


@app.route('/api/generate/single', methods=['POST'])
def generate_single():
    """Generate a single diffraction pattern"""
    data = request.json
    
    # Extract parameters with defaults
    material = data.get('material', 'Al')
    zone_axis = data.get('zone_axis', [1, 1, 1])
    voltage = data.get('voltage', 200)
    size = data.get('size', 256)

    # Sample & Physics
    sample_thickness = data.get('sample_thickness', None)
    max_reflection_index = data.get('max_reflection_index', 6)

    # Noise & Detector
    noise_level = data.get('noise_level', 0.02)
    background_variation = data.get('background_variation', 0.01)
    detector_defects = data.get('detector_defects', True)
    detector_response = data.get('detector_response', 0.9)
    radial_decay_factor = data.get('radial_decay_factor', 3.0)
    logarithmic_scaling = data.get('logarithmic_scaling', 5.0)

    # Beam & Optics
    camera_length = data.get('camera_length', 300)
    convergence_angle = data.get('convergence_angle', 1.5)

    # Intensity
    direct_beam_intensity = data.get('direct_beam_intensity', 0.7)
    spot_intensity_factor = data.get('spot_intensity_factor', 0.4)
    spot_size_factor = data.get('spot_size_factor', 1.2)

    # Physical
    temperature_factor = data.get('temperature_factor', 0.8)
    deviation_parameter = data.get('deviation_parameter', 0.1)

    # Features
    kikuchi_intensity_factor = data.get('kikuchi_intensity_factor', 0.03)
    kikuchi_probability = data.get('kikuchi_probability', 0.3)
    holz_intensity = data.get('holz_intensity', 0.02)
    holz_probability = data.get('holz_probability', 0.2)

    # Mask
    generate_mask = data.get('generate_mask', True)
    mask_type = data.get('mask_type', 'binary')
    mask_visibility_threshold = data.get('mask_visibility_threshold', 0.15)

    try:
        # Generate pattern with all parameters
        pattern, mask = generator.generate(
            material=material,
            zone_axis=zone_axis,
            voltage=voltage,
            size=size,
            sample_thickness=sample_thickness,
            max_reflection_index=max_reflection_index,
            noise_level=noise_level,
            background_variation=background_variation,
            detector_defects=detector_defects,
            detector_response=detector_response,
            radial_decay_factor=radial_decay_factor,
            logarithmic_scaling=logarithmic_scaling,
            temperature_factor=temperature_factor,
            direct_beam_intensity=direct_beam_intensity,
            spot_intensity_factor=spot_intensity_factor,
            kikuchi_intensity_factor=kikuchi_intensity_factor,
            kikuchi_probability=kikuchi_probability,
            holz_intensity=holz_intensity,
            holz_probability=holz_probability,
            spot_size_factor=spot_size_factor,
            camera_length=camera_length,
            convergence_angle=convergence_angle,
            deviation_parameter=deviation_parameter,
            generate_mask=generate_mask,
            mask_type=mask_type,
            mask_visibility_threshold=mask_visibility_threshold
        )

        # Convert to base64
        pattern_b64 = numpy_to_base64(pattern)
        mask_b64 = numpy_to_base64(mask) if mask is not None else None

        return jsonify({
            'success': True,
            'pattern': pattern_b64,
            'mask': mask_b64,
            'parameters': {
                'material': material,
                'zone_axis': zone_axis,
                'voltage': voltage,
                'size': size
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/generate/batch', methods=['POST'])
def generate_batch():
    """Generate a batch dataset of diffraction patterns"""
    data = request.json

    # Extract parameters
    materials = data.get('materials', ['Al'])
    num_patterns = data.get('num_patterns_per_material', 100)
    output_dir = data.get('output_dir', 'dataset')
    voltage_range = data.get('voltage_range', [100, 300])
    size = data.get('size', 256)
    mask_type = data.get('mask_type', 'binary')

    # Extract advanced parameters - convert single values to ranges
    def to_range(val, default_range):
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return tuple(val)
        elif val is not None:
            # Create a small range around the value
            variation = val * 0.1  # 10% variation
            return (val - variation, val + variation)
        return default_range

    noise_level = to_range(data.get('noise_level'), (0.01, 0.03))
    background_variation = to_range(data.get('background_variation'), (0.005, 0.02))
    detector_defects = data.get('detector_defects', True)
    temperature_factor = to_range(data.get('temperature_factor'), (0.7, 0.9))
    direct_beam_intensity = to_range(data.get('direct_beam_intensity'), (0.6, 0.8))
    spot_intensity_factor = to_range(data.get('spot_intensity_factor'), (0.3, 0.5))
    kikuchi_intensity_factor = to_range(data.get('kikuchi_intensity_factor'), (0.01, 0.05))
    holz_intensity = to_range(data.get('holz_intensity'), (0.01, 0.04))
    spot_size_factor = to_range(data.get('spot_size_factor'), (1.0, 1.5))
    camera_length = to_range(data.get('camera_length'), (250, 400))
    convergence_angle = to_range(data.get('convergence_angle'), (1.0, 2.0))
    deviation_parameter = to_range(data.get('deviation_parameter'), (0.08, 0.15))

    try:
        # Generate dataset with all parameters
        stats = generator.generate_dataset(
            materials=materials,
            num_patterns_per_material=num_patterns,
            output_dir=output_dir,
            voltage_range=tuple(voltage_range),
            size=size,
            mask_type=mask_type,
            noise_level=noise_level,
            background_variation=background_variation,
            detector_defects=detector_defects,
            temperature_factor=temperature_factor,
            direct_beam_intensity=direct_beam_intensity,
            spot_intensity_factor=spot_intensity_factor,
            kikuchi_intensity_factor=kikuchi_intensity_factor,
            holz_intensity=holz_intensity,
            spot_size_factor=spot_size_factor,
            camera_length=camera_length,
            convergence_angle=convergence_angle,
            deviation_parameter=deviation_parameter
        )

        return jsonify({
            'success': True,
            'stats': stats,
            'output_dir': output_dir
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/preview', methods=['POST'])
def preview_pattern():
    """Generate a quick preview (low resolution for speed)"""
    data = request.json

    material = data.get('material', 'Al')
    zone_axis = data.get('zone_axis', [1, 1, 1])
    voltage = data.get('voltage', 200)

    try:
        # Generate small preview
        pattern, _ = generator.generate(
            material=material,
            zone_axis=zone_axis,
            voltage=voltage,
            size=128,  # Smaller for speed
            noise_level=0.01,
            generate_mask=False
        )

        # Convert to base64
        pattern_b64 = numpy_to_base64(pattern)

        return jsonify({
            'success': True,
            'preview': pattern_b64
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/export/pattern', methods=['POST'])
def export_pattern():
    """Export pattern to file"""
    data = request.json

    # Generate pattern
    material = data.get('material', 'Al')
    zone_axis = data.get('zone_axis', [1, 1, 1])
    voltage = data.get('voltage', 200)
    size = data.get('size', 256)

    pattern, mask = generator.generate(
        material=material,
        zone_axis=zone_axis,
        voltage=voltage,
        size=size,
        generate_mask=True
    )

    # Save to file
    output_dir = Path('exports')
    output_dir.mkdir(exist_ok=True)

    filename = f"{material}_{''.join(map(str, zone_axis))}_{voltage}kV"

    # Save pattern
    pattern_path = output_dir / f"{filename}_pattern.png"
    Image.fromarray((pattern * 255).astype(np.uint8)).save(pattern_path)

    # Save mask
    mask_path = output_dir / f"{filename}_mask.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

    return jsonify({
        'success': True,
        'files': {
            'pattern': str(pattern_path),
            'mask': str(mask_path)
        }
    })


@app.route('/api/create/animation', methods=['POST'])
def create_animation():
    """Create diffraction animation"""
    data = request.json

    material = data.get('material', 'Al')
    zone_axis = data.get('zone_axis', [0, 0, 1])
    voltage = data.get('voltage', 200)
    duration = data.get('duration', 10)
    fps = data.get('fps', 30)

    try:
        from animator import DiffractionAnimator

        animator = DiffractionAnimator()

        # Create output directory
        output_dir = Path('animations')
        output_dir.mkdir(exist_ok=True)

        filename = f"{material}_{''.join(map(str, zone_axis))}_{voltage}kV.gif"
        save_path = output_dir / filename

        # Create animation
        animator.create_animation(
            material=material,
            zone_axis=zone_axis,
            voltage=voltage,
            duration=duration,
            fps=fps,
            save_path=str(save_path)
        )

        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(save_path)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/animations/<filename>')
def serve_animation(filename):
    """Serve animation files"""
    try:
        animations_dir = Path('animations')
        return send_file(animations_dir / filename, mimetype='image/gif')
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/')
def index():
    """Serve the main GUI"""
    return send_file('diffraction_app.html')


if __name__ == '__main__':
    print("🔬 STEM Diffraction Pattern Generator")
    print("=" * 50)
    print(f"Available materials: {', '.join(generator.get_available_materials())}")
    print("\n🌐 Starting server at http://localhost:5000")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)