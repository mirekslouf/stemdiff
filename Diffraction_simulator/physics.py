"""
Physics Engine for STEM Diffraction Pattern Generation
Handles all electron diffraction physics calculations
"""

import numpy as np
from typing import Dict, List, Tuple


class DiffractionPhysics:
    """Core physics calculations for electron diffraction"""
    
    # Physical constants
    M0 = 9.1094e-31  # electron rest mass (kg)
    E = 1.6022e-19   # elementary charge (C)
    H = 6.6261e-34   # Planck's constant (J·s)
    C = 2.9979e8     # speed of light (m/s)
    
    @staticmethod
    def calculate_wavelength(voltage_kv: float) -> float:
        """
        Calculate electron wavelength from accelerating voltage
        
        Args:
            voltage_kv: Voltage in kilovolts
            
        Returns:
            Wavelength in Angstroms
        """
        V = voltage_kv * 1000  # Convert to volts
        wavelength = (
            DiffractionPhysics.H / 
            np.sqrt(2 * DiffractionPhysics.M0 * DiffractionPhysics.E * V *
                   (1 + DiffractionPhysics.E * V / 
                    (2 * DiffractionPhysics.M0 * DiffractionPhysics.C ** 2)))
        ) * 1e10  # Convert to Angstroms
        
        return wavelength
    
    @staticmethod
    def calculate_reciprocal_lattice(lattice_params: Dict) -> Tuple:
        """
        Calculate reciprocal lattice parameters
        
        Args:
            lattice_params: Dict with keys 'a', 'b', 'c', 'alpha', 'beta', 'gamma'
            
        Returns:
            Tuple of (a_recip, b_recip, c_recip, alpha_recip, beta_recip, gamma_recip)
        """
        a = lattice_params['a']
        b = lattice_params['b']
        c = lattice_params['c']
        alpha = np.radians(lattice_params['alpha'])
        beta = np.radians(lattice_params['beta'])
        gamma = np.radians(lattice_params['gamma'])
        
        # Unit cell volume
        volume = a * b * c * np.sqrt(
            1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2
            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
        
        # Reciprocal lattice vectors
        a_recip = 2 * np.pi * b * c * np.sin(alpha) / volume
        b_recip = 2 * np.pi * a * c * np.sin(beta) / volume
        c_recip = 2 * np.pi * a * b * np.sin(gamma) / volume
        
        alpha_recip = np.arccos(
            (np.cos(beta) * np.cos(gamma) - np.cos(alpha)) /
            (np.sin(beta) * np.sin(gamma))
        )
        beta_recip = np.arccos(
            (np.cos(alpha) * np.cos(gamma) - np.cos(beta)) /
            (np.sin(alpha) * np.sin(gamma))
        )
        gamma_recip = np.arccos(
            (np.cos(alpha) * np.cos(beta) - np.cos(gamma)) /
            (np.sin(alpha) * np.sin(beta))
        )
        
        return (a_recip, b_recip, c_recip, alpha_recip, beta_recip, gamma_recip)
    
    @staticmethod
    def is_allowed_reflection(h: int, k: int, l: int, structure: str) -> bool:
        """
        Check if reflection is allowed based on systematic absences
        
        Args:
            h, k, l: Miller indices
            structure: Crystal structure type
            
        Returns:
            True if reflection is allowed
        """
        if structure == 'fcc':
            return h % 2 == k % 2 == l % 2
        elif structure == 'bcc':
            return (h + k + l) % 2 == 0
        elif structure == 'diamond':
            if h % 2 == k % 2 == l % 2 == 0:
                return (h + k + l) % 4 == 0
            else:
                return h % 2 == k % 2 == l % 2 == 1
        elif structure == 'zincblende':
            if h % 2 == k % 2 == l % 2 == 0:
                return (h + k + l) % 4 == 0
            elif h % 2 == k % 2 == l % 2 == 1:
                return (h + k + l) % 2 == 1
            else:
                return False
        elif structure == 'perovskite':
            return h % 2 == k % 2 == l % 2
        elif structure == 'wurtzite' or structure == 'hcp':
            if l % 2 == 0:
                return True
            else:
                return (h + 2 * k) % 3 != 0
        else:
            return True  # Allow all for unknown structures
    
    @staticmethod
    def calculate_structure_factor(
        h: int, k: int, l: int,
        atomic_positions: List[Dict],
        scattering_factors: Dict[str, float]
    ) -> complex:
        """
        Calculate structure factor for a reflection
        
        Args:
            h, k, l: Miller indices
            atomic_positions: List of dicts with 'element' and 'position' keys
            scattering_factors: Dict mapping element symbols to scattering factors
            
        Returns:
            Complex structure factor
        """
        structure_factor = 0 + 0j
        
        for atom in atomic_positions:
            element = atom['element']
            if element not in scattering_factors:
                continue
                
            f = scattering_factors[element]
            px, py, pz = atom['position']
            phase = 2 * np.pi * (h * px + k * py + l * pz)
            structure_factor += f * np.exp(1j * phase)
        
        return structure_factor
    
    @staticmethod
    def calculate_intensity(
        structure_factor: complex,
        g_magnitude: float,
        wavelength: float,
        sample_thickness: float,
        deviation_param: float,
        temperature_factor: float
    ) -> float:
        """
        Calculate diffraction intensity using kinematical approximation
        
        Args:
            structure_factor: Complex structure factor
            g_magnitude: Magnitude of reciprocal lattice vector (1/Angstrom)
            wavelength: Electron wavelength (Angstrom)
            sample_thickness: Sample thickness (nm)
            deviation_param: Deviation parameter (1/nm)
            temperature_factor: Debye-Waller factor (0-1)
            
        Returns:
            Intensity value
        """
        sf_abs = abs(structure_factor)
        epsilon = 1e-10
        
        # Calculate extinction distance
        denom = wavelength * g_magnitude * sf_abs
        if denom < epsilon:
            return 0.0
        
        extinction_distance = 1.0 / denom
        
        # Calculate sin term
        sin_term = np.sin(np.pi * sample_thickness / extinction_distance)
        
        # Protect against division by zero
        s = deviation_param
        pi_s = np.pi * s
        if abs(pi_s) < epsilon:
            return 0.0
        
        # Calculate intensity
        intensity = (sf_abs ** 2) * (sin_term ** 2) / (pi_s ** 2)
        
        # Apply Debye-Waller temperature factor
        B = -np.log(max(temperature_factor, epsilon)) * 8 * np.pi ** 2
        intensity *= np.exp(-B * (g_magnitude / (4 * np.pi)) ** 2)
        
        return float(np.abs(intensity))
