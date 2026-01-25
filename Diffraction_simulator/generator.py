"""
Diffraction Pattern Generator
Clean, modular API for generating STEM diffraction patterns
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
import cv2

from physics import DiffractionPhysics


class DiffractionGenerator:
    """
    Simplified STEM diffraction pattern generator
    
    Usage:
        # Create generator
        gen = DiffractionGenerator()
        
        # Generate single pattern
        pattern, mask = gen.generate(
            material='Al',
            zone_axis=[1, 1, 1],
            voltage=200
        )
        
        # Generate dataset
        gen.generate_dataset(
            materials=['Al', 'Cu', 'Si'],
            num_patterns_per_material=100,
            output_dir='dataset'
        )
    """
    
    def __init__(self, crystals_db_path: str = 'crystals.json'):
        """Initialize generator with crystal database"""
        self.crystals_db = self._load_crystals_db(crystals_db_path)
        self.physics = DiffractionPhysics()
        
    def _load_crystals_db(self, path: str) -> Dict:
        """Load crystal structure database"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_available_materials(self) -> List[str]:
        """Get list of all available materials"""
        materials = []
        materials.extend(self.crystals_db['simple_materials'].keys())
        materials.extend(self.crystals_db['complex_materials'].keys())
        return sorted(materials)
    
    def get_material_info(self, material: str) -> Dict:
        """Get information about a material"""
        if material in self.crystals_db['simple_materials']:
            return self.crystals_db['simple_materials'][material]
        elif material in self.crystals_db['complex_materials']:
            return self.crystals_db['complex_materials'][material]
        else:
            raise ValueError(f"Material {material} not found in database")
    
    def generate(
        self,
        material: str,
        zone_axis: List[int],
        voltage: float = 200,
        size: int = 256,
        sample_thickness: Optional[float] = None,
        max_reflection_index: int = 6,
        noise_level: float = 0.02,
        background_variation: float = 0.01,
        detector_defects: bool = True,
        detector_response: float = 0.9,
        radial_decay_factor: float = 3.0,
        logarithmic_scaling: float = 5.0,
        temperature_factor: float = 0.8,
        direct_beam_intensity: float = 0.7,
        spot_intensity_factor: float = 0.4,
        kikuchi_intensity_factor: float = 0.03,
        kikuchi_probability: float = 0.3,
        holz_intensity: float = 0.02,
        holz_probability: float = 0.2,
        spot_size_factor: float = 1.2,
        camera_length: float = 300,
        convergence_angle: float = 1.5,
        deviation_parameter: float = 0.1,
        generate_mask: bool = True,
        mask_type: str = 'binary',
        mask_visibility_threshold: float = 0.15
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate a single diffraction pattern

        Args:
            material: Material name (e.g., 'Al', 'Cu', 'Si')
            zone_axis: Zone axis direction [h, k, l]
            voltage: Accelerating voltage in kV
            size: Image size in pixels
            sample_thickness: Sample thickness in nm (if None, uses default)
            max_reflection_index: Maximum h,k,l values for reflections (higher = more spots, slower)
            noise_level: Amount of noise to add (0-1)
            background_variation: Level of non-uniform background (0-1)
            detector_defects: Whether to add detector defects
            detector_response: Power law exponent for detector (1.0 = linear)
            radial_decay_factor: Radial intensity fall-off factor
            logarithmic_scaling: Dynamic range compression factor
            temperature_factor: Debye-Waller factor (0-1)
            direct_beam_intensity: Intensity factor for direct beam (0-1)
            spot_intensity_factor: Scaling factor for diffraction spots (0-1)
            kikuchi_intensity_factor: Intensity factor for Kikuchi lines (0-1)
            kikuchi_probability: Probability of Kikuchi lines appearing (0-1)
            holz_intensity: Intensity factor for HOLZ rings (0-1)
            holz_probability: Probability of HOLZ rings appearing (0-1)
            spot_size_factor: Controls size/sharpness of spots
            camera_length: Camera length in mm
            convergence_angle: Beam convergence semi-angle in mrad
            deviation_parameter: Excitation error in 1/nm
            generate_mask: Whether to generate segmentation mask
            mask_type: Type of mask ('binary', 'intensity', or 'adaptive')
            mask_visibility_threshold: Minimum intensity for spot to be masked (0-1)

        Returns:
            (pattern, mask) tuple. mask is None if generate_mask=False
        """
        # Get material properties
        mat_info = self.get_material_info(material)

        if sample_thickness is None:
            sample_thickness = mat_info['default_thickness']

        # Calculate physics parameters
        wavelength = self.physics.calculate_wavelength(voltage)

        recip_params = self.physics.calculate_reciprocal_lattice(mat_info['lattice'])
        a_recip, b_recip, c_recip = recip_params[:3]

        # Generate reflections
        reflections = self._generate_reflections(
            mat_info=mat_info,
            zone_axis=zone_axis,
            a_recip=a_recip,
            b_recip=b_recip,
            c_recip=c_recip,
            wavelength=wavelength,
            sample_thickness=sample_thickness,
            deviation_parameter=deviation_parameter,
            temperature_factor=temperature_factor,
            max_index=max_reflection_index
        )

        # Create diffraction pattern image with all features
        # Apply probabilities to Kikuchi and HOLZ
        actual_kikuchi_intensity = kikuchi_intensity_factor if np.random.random() < kikuchi_probability else 0
        actual_holz_intensity = holz_intensity if np.random.random() < holz_probability else 0

        pattern = self._create_pattern_image(
            reflections=reflections,
            zone_axis=zone_axis,
            size=size,
            wavelength=wavelength,
            a_recip=a_recip,
            b_recip=b_recip,
            c_recip=c_recip,
            camera_length=camera_length,
            convergence_angle=convergence_angle,
            direct_beam_intensity=direct_beam_intensity,
            spot_intensity_factor=spot_intensity_factor,
            spot_size_factor=spot_size_factor,
            kikuchi_intensity_factor=actual_kikuchi_intensity,
            holz_intensity=actual_holz_intensity
        )

        # Add noise and detector effects
        pattern = self._add_noise_and_effects(
            pattern, noise_level, background_variation, detector_defects,
            detector_response, radial_decay_factor, logarithmic_scaling
        )

        # Generate mask if requested (uses reflection intensities)
        mask = None
        if generate_mask:
            mask = self._create_mask(
                reflections=reflections,
                zone_axis=zone_axis,
                size=size,
                wavelength=wavelength,
                a_recip=a_recip,
                b_recip=b_recip,
                c_recip=c_recip,
                camera_length=camera_length,
                spot_size_factor=spot_size_factor,
                mask_type=mask_type,
                intensity_threshold=mask_visibility_threshold
            )

        return pattern, mask

    def _generate_reflections(
        self,
        mat_info: Dict,
        zone_axis: List[int],
        a_recip: float,
        b_recip: float,
        c_recip: float,
        wavelength: float,
        sample_thickness: float,
        deviation_parameter: float,
        temperature_factor: float,
        max_index: int = 6
    ) -> List[Dict]:
        """Generate allowed reflections for the given conditions"""
        reflections = []

        # Normalize zone axis
        zone_norm = np.sqrt(sum(z**2 for z in zone_axis))
        uz = [z / zone_norm for z in zone_axis]

        structure = mat_info['structure']
        atomic_positions = mat_info['atomic_positions']
        scattering_factors = self.crystals_db['scattering_factors']

        for h in range(-max_index, max_index + 1):
            for k in range(-max_index, max_index + 1):
                for l in range(-max_index, max_index + 1):
                    if h == 0 and k == 0 and l == 0:
                        continue

                    # Check systematic absences
                    if not self.physics.is_allowed_reflection(h, k, l, structure):
                        continue

                    # Calculate g-vector magnitude
                    g_magnitude = np.sqrt(
                        (h * a_recip)**2 + (k * b_recip)**2 + (l * c_recip)**2
                    )

                    # Check if reflection is near zone axis
                    dot_product = h * uz[0] + k * uz[1] + l * uz[2]
                    if abs(dot_product) > 0.1:
                        continue

                    # Calculate structure factor
                    structure_factor = self.physics.calculate_structure_factor(
                        h, k, l, atomic_positions, scattering_factors
                    )

                    # Calculate intensity using provided deviation parameter
                    dev_param = deviation_parameter * (abs(h) + abs(k) + abs(l)) / max(g_magnitude, 1e-10)

                    intensity = self.physics.calculate_intensity(
                        structure_factor=structure_factor,
                        g_magnitude=g_magnitude,
                        wavelength=wavelength,
                        sample_thickness=sample_thickness,
                        deviation_param=dev_param,
                        temperature_factor=temperature_factor
                    )

                    if intensity > 1e-6:
                        reflections.append({
                            'h': h, 'k': k, 'l': l,
                            'intensity': intensity / 100,  # Scale down
                            'g_magnitude': g_magnitude
                        })

        return sorted(reflections, key=lambda x: x['intensity'], reverse=True)[:100]

    def _create_pattern_image(
        self,
        reflections: List[Dict],
        zone_axis: List[int],
        size: int,
        wavelength: float,
        a_recip: float,
        b_recip: float,
        c_recip: float,
        camera_length: float,
        convergence_angle: float,
        direct_beam_intensity: float,
        spot_intensity_factor: float,
        spot_size_factor: float,
        kikuchi_intensity_factor: float,
        holz_intensity: float
    ) -> np.ndarray:
        """Create the diffraction pattern image with all features"""
        pattern = np.zeros((size, size))
        center_x, center_y = size // 2, size // 2

        # Create coordinate grid
        x, y = np.meshgrid(np.arange(size), np.arange(size))

        # Add direct beam
        beam_radius = size * convergence_angle / (25 * wavelength)
        direct_beam = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * beam_radius**2))
        pattern += direct_beam * direct_beam_intensity

        # Calculate perpendicular vectors to zone axis
        if abs(zone_axis[2]) < 0.9:
            u1 = np.cross(zone_axis, [0, 0, 1])
        else:
            u1 = np.cross(zone_axis, [1, 0, 0])
        u1 = u1 / np.linalg.norm(u1)

        u2 = np.cross(zone_axis, u1)
        u2 = u2 / np.linalg.norm(u2)

        # Scale factor for positioning
        scale_factor = size / (4 * wavelength * camera_length)
        spot_size = size / 200 * spot_size_factor

        # Add diffraction spots
        for ref in reflections:
            h, k, l = ref['h'], ref['k'], ref['l']
            intensity = ref['intensity'] * spot_intensity_factor

            # Project g-vector onto diffraction plane
            g_vector = [h * a_recip, k * b_recip, l * c_recip]
            proj_u1 = np.dot(g_vector, u1)
            proj_u2 = np.dot(g_vector, u2)

            # Convert to pixel coordinates
            spot_x = center_x + proj_u1 * scale_factor
            spot_y = center_y + proj_u2 * scale_factor

            # Check boundaries
            if 0 <= spot_x < size and 0 <= spot_y < size:
                r = np.sqrt((x - spot_x)**2 + (y - spot_y)**2)

                # Realistic spot profile
                spot_core = np.exp(-r**2 / (2 * (spot_size * 0.5)**2))
                spot_tail = 1 / (1 + (r / spot_size)**4)
                spot = 0.8 * spot_core + 0.2 * spot_tail

                pattern += spot * intensity

        # Add Kikuchi lines (simplified version)
        if kikuchi_intensity_factor > 0 and np.random.random() < 0.3:
            for ref in reflections[:5]:  # Use strongest reflections
                h, k, l = ref['h'], ref['k'], ref['l']
                g_vector = [h * a_recip, k * b_recip, l * c_recip]
                g_magnitude = np.sqrt(sum(g**2 for g in g_vector))

                if g_magnitude > 1e-6:
                    # Simplified Kikuchi line rendering
                    proj_u1 = np.dot(g_vector, u1)
                    proj_u2 = np.dot(g_vector, u2)

                    # Create line mask
                    line_mask = np.zeros((size, size))
                    angle = np.arctan2(proj_u2, proj_u1)

                    for i in range(size):
                        for j in range(size):
                            dist = abs((j - center_y) * np.cos(angle) - (i - center_x) * np.sin(angle))
                            if dist < 2:
                                line_mask[j, i] = np.exp(-dist**2 / 2)

                    line_mask = ndimage.gaussian_filter(line_mask, sigma=size / 100)
                    pattern += line_mask * kikuchi_intensity_factor * ref['intensity']

        # Add HOLZ rings
        if holz_intensity > 0 and np.random.random() < 0.2:
            holz_radius = size * wavelength * 4 / 4
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            holz_ring = np.exp(-(r - holz_radius)**2 / (2 * (size / 50)**2))
            pattern += holz_ring * holz_intensity

        return pattern

    def _add_noise_and_effects(
        self,
        pattern: np.ndarray,
        noise_level: float,
        background_variation: float,
        detector_defects: bool,
        detector_response: float,
        radial_decay_factor: float,
        logarithmic_scaling: float
    ) -> np.ndarray:
        """Add realistic noise and detector effects with full parameter control"""
        size = pattern.shape[0]

        # Add non-uniform background
        y_gradient, x_gradient = np.meshgrid(
            np.linspace(-1, 1, size),
            np.linspace(-1, 1, size)
        )
        background = background_variation * (
            1 + np.sin(x_gradient * np.pi * 1.5) * 0.5 +
            np.cos(y_gradient * np.pi * 2.3) * 0.5
        )
        background = ndimage.gaussian_filter(background, sigma=size / 10)
        pattern += background

        # Add detector defects if enabled
        if detector_defects:
            # Dead pixels
            num_dead_pixels = int(size * size * 0.0005)
            for _ in range(num_dead_pixels):
                px, py = np.random.randint(0, size, 2)
                pattern[py, px] = 0

            # Hot pixels
            num_hot_pixels = int(size * size * 0.0002)
            for _ in range(num_hot_pixels):
                px, py = np.random.randint(0, size, 2)
                pattern[py, px] = 1

            # Clusters of bad pixels
            num_clusters = np.random.randint(1, 3)
            for _ in range(num_clusters):
                cx, cy = np.random.randint(0, size, 2)
                cluster_size = np.random.randint(2, 5)
                cluster_value = np.random.choice([0, 1])

                for i in range(-cluster_size // 2, cluster_size // 2 + 1):
                    for j in range(-cluster_size // 2, cluster_size // 2 + 1):
                        if 0 <= cx + i < size and 0 <= cy + j < size:
                            if np.random.random() < 0.6:
                                pattern[cy + j, cx + i] = cluster_value

        # Add Poisson noise
        signal_level = np.mean(pattern) * 500
        signal_level = max(signal_level, 1e-10)
        pattern = np.random.poisson(pattern * signal_level) / signal_level

        # Add readout noise
        pattern += np.random.normal(0, noise_level, pattern.shape)

        # Apply detector response (power law)
        epsilon = 1e-10
        pattern = np.power(np.maximum(pattern, epsilon), detector_response)

        # Apply radial intensity decay
        center_x, center_y = size // 2, size // 2
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        radial_decay = np.exp(-r / (size / radial_decay_factor))
        pattern *= radial_decay

        # Normalize and clip
        pattern = np.clip(pattern, 0, 1)

        # Logarithmic intensity scaling (dynamic range compression)
        pattern = np.log1p(pattern * logarithmic_scaling) / np.log(logarithmic_scaling + 1)

        # Final normalization
        if np.max(pattern) > 0:
            pattern = pattern / np.max(pattern)

        return pattern

    def _create_mask(
        self,
        reflections: List[Dict],
        zone_axis: List[int],
        size: int,
        wavelength: float,
        a_recip: float,
        b_recip: float,
        c_recip: float,
        camera_length: float,
        spot_size_factor: float,
        mask_type: str = 'binary',
        intensity_threshold: float = 0.15
    ) -> np.ndarray:
        """
        Create mask using geometric spot positions
        Only includes spots with intensity above threshold
        """
        mask = np.zeros((size, size), dtype=np.uint8)
        center_x, center_y = size // 2, size // 2

        # Calculate perpendicular vectors
        if abs(zone_axis[2]) < 0.9:
            u1 = np.cross(zone_axis, [0, 0, 1])
        else:
            u1 = np.cross(zone_axis, [1, 0, 0])
        u1 = u1 / np.linalg.norm(u1)

        u2 = np.cross(zone_axis, u1)
        u2 = u2 / np.linalg.norm(u2)

        # Use actual camera_length
        scale_factor = size / (4 * wavelength * camera_length)

        # Spot size
        spot_size = int(np.ceil(size / 200 * spot_size_factor * 1.5))
        spot_size = max(spot_size, 3)

        # Add direct beam (always visible)
        cv2.circle(mask, (center_x, center_y), spot_size, 255, -1, cv2.LINE_AA)

        # Add diffraction spots ONLY if intensity above threshold
        for ref in reflections:
            h, k, l = ref['h'], ref['k'], ref['l']
            intensity = ref['intensity']

            # Skip if intensity too low
            if intensity < intensity_threshold:
                continue

            # Project g-vector
            g_vector = [h * a_recip, k * b_recip, l * c_recip]
            proj_u1 = np.dot(g_vector, u1)
            proj_u2 = np.dot(g_vector, u2)

            # Convert to pixel coordinates
            spot_x = int(round(center_x + proj_u1 * scale_factor))
            spot_y = int(round(center_y + proj_u2 * scale_factor))

            # Check boundaries
            if 0 <= spot_x < size and 0 <= spot_y < size:
                if mask_type == 'binary':
                    # Binary: all spots white
                    cv2.circle(mask, (spot_x, spot_y), spot_size, 255, -1, cv2.LINE_AA)

                elif mask_type == 'intensity':
                    # Intensity-weighted
                    intensity_value = min(255, int(intensity * 1000))
                    cv2.circle(mask, (spot_x, spot_y), spot_size, intensity_value, -1, cv2.LINE_AA)

                elif mask_type == 'adaptive':
                    # Adaptive size based on intensity
                    adaptive_size = max(2, int(spot_size * (0.5 + intensity * 50)))
                    cv2.circle(mask, (spot_x, spot_y), adaptive_size, 255, -1, cv2.LINE_AA)

        # Convert to float
        mask = mask.astype(np.float32) / 255.0

        return mask

    def generate_dataset(
        self,
        materials: List[str],
        num_patterns_per_material: int,
        output_dir: str,
        voltage_range: Tuple[float, float] = (100, 300),
        size: int = 256,
        mask_type: str = 'binary',
        # Advanced parameters
        noise_level: Tuple[float, float] = (0.01, 0.03),
        background_variation: Tuple[float, float] = (0.005, 0.02),
        detector_defects: bool = True,
        temperature_factor: Tuple[float, float] = (0.7, 0.9),
        direct_beam_intensity: Tuple[float, float] = (0.6, 0.8),
        spot_intensity_factor: Tuple[float, float] = (0.3, 0.5),
        kikuchi_intensity_factor: Tuple[float, float] = (0.01, 0.05),
        holz_intensity: Tuple[float, float] = (0.01, 0.04),
        spot_size_factor: Tuple[float, float] = (1.0, 1.5),
        camera_length: Tuple[float, float] = (250, 400),
        convergence_angle: Tuple[float, float] = (1.0, 2.0),
        deviation_parameter: Tuple[float, float] = (0.08, 0.15)
    ) -> Dict:
        """
        Generate a complete dataset of diffraction patterns

        Args:
            materials: List of material names
            num_patterns_per_material: Number of patterns to generate per material
            output_dir: Output directory path
            voltage_range: (min, max) voltage in kV
            size: Image size in pixels
            mask_type: Type of mask to generate
            noise_level: Range for noise level (min, max)
            background_variation: Range for background variation
            detector_defects: Whether to include detector defects
            temperature_factor: Range for temperature factor
            direct_beam_intensity: Range for direct beam intensity
            spot_intensity_factor: Range for spot intensity
            kikuchi_intensity_factor: Range for Kikuchi intensity
            holz_intensity: Range for HOLZ intensity
            spot_size_factor: Range for spot size
            camera_length: Range for camera length (mm)
            convergence_angle: Range for convergence angle (mrad)
            deviation_parameter: Range for deviation parameter

        Returns:
            Dictionary with statistics about the generated dataset
        """
        import os
        from PIL import Image

        output_path = Path(output_dir)
        images_path = output_path / 'images'
        masks_path = output_path / 'masks'

        images_path.mkdir(parents=True, exist_ok=True)
        masks_path.mkdir(parents=True, exist_ok=True)

        stats = {
            'total_patterns': 0,
            'materials': {},
            'parameters': {
                'size': size,
                'voltage_range': voltage_range,
                'mask_type': mask_type
            }
        }

        for material in materials:
            mat_info = self.get_material_info(material)
            zone_axes = mat_info['common_zone_axes']

            stats['materials'][material] = 0

            for i in range(num_patterns_per_material):
                # Randomize parameters within ranges
                zone_axis = zone_axes[i % len(zone_axes)]

                # Helper function to randomize parameter
                def rand_param(param_range):
                    if isinstance(param_range, tuple):
                        return np.random.uniform(param_range[0], param_range[1])
                    return param_range

                # Generate pattern with randomized parameters
                pattern, mask = self.generate(
                    material=material,
                    zone_axis=zone_axis,
                    voltage=rand_param(voltage_range),
                    size=size,
                    noise_level=rand_param(noise_level),
                    background_variation=rand_param(background_variation),
                    detector_defects=detector_defects,
                    temperature_factor=rand_param(temperature_factor),
                    direct_beam_intensity=rand_param(direct_beam_intensity),
                    spot_intensity_factor=rand_param(spot_intensity_factor),
                    kikuchi_intensity_factor=rand_param(kikuchi_intensity_factor),
                    holz_intensity=rand_param(holz_intensity),
                    spot_size_factor=rand_param(spot_size_factor),
                    camera_length=rand_param(camera_length),
                    convergence_angle=rand_param(convergence_angle),
                    deviation_parameter=rand_param(deviation_parameter),
                    generate_mask=True,
                    mask_type=mask_type
                )

                # Save files
                filename = f"{material}_{i:04d}"

                # Save pattern
                pattern_img = (pattern * 255).astype(np.uint8)
                Image.fromarray(pattern_img).save(images_path / f"{filename}.png")

                # Save mask
                mask_img = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_img).save(masks_path / f"{filename}.png")

                stats['total_patterns'] += 1
                stats['materials'][material] += 1

        # Save metadata
        import json
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return stats


if __name__ == '__main__':
    # Example usage
    gen = DiffractionGenerator()

    print("Available materials:")
    for mat in gen.get_available_materials():
        info = gen.get_material_info(mat)
        print(f"  {mat}: {info['name']}")

    # Generate a single pattern
    pattern, mask = gen.generate('Al', [1, 1, 1], voltage=200)
    print(f"\nGenerated pattern shape: {pattern.shape}")
    print(f"Generated mask shape: {mask.shape}")