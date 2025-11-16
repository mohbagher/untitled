"""
Unit Tests for Electromagnetic Calculations

This module contains unit tests for the functions in src/calculations.py
"""

import unittest
import numpy as np
from scipy import constants
import sys
import os

# Add parent directory to path to import src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.calculations import (
    coulomb_force,
    electric_field_point_charge,
    electric_field_multiple_charges,
    electric_potential,
    magnetic_field_wire,
    wavelength_from_frequency,
    frequency_from_wavelength,
    free_space_path_loss,
    dipole_radiation_pattern
)


class TestCoulombForce(unittest.TestCase):
    """Test cases for Coulomb force calculations."""
    
    def test_repulsive_force(self):
        """Test force between two positive charges."""
        q1 = 1e-6  # 1 μC
        q2 = 1e-6  # 1 μC
        r = 1.0    # 1 meter
        
        F, force_type = coulomb_force(q1, q2, r)
        
        self.assertEqual(force_type, 'repulsive')
        self.assertGreater(F, 0)
        # Expected: k * 1e-6 * 1e-6 / 1^2 ≈ 8.99 mN
        self.assertAlmostEqual(F, 8.99e-3, places=5)
    
    def test_attractive_force(self):
        """Test force between opposite charges."""
        q1 = 1e-6   # 1 μC
        q2 = -1e-6  # -1 μC
        r = 1.0     # 1 meter
        
        F, force_type = coulomb_force(q1, q2, r)
        
        self.assertEqual(force_type, 'attractive')
        self.assertGreater(F, 0)
    
    def test_inverse_square_law(self):
        """Test that force follows inverse square law."""
        q1 = 1e-6
        q2 = 1e-6
        r1 = 1.0
        r2 = 2.0
        
        F1, _ = coulomb_force(q1, q2, r1)
        F2, _ = coulomb_force(q1, q2, r2)
        
        # F2 should be 1/4 of F1 (since r2 = 2*r1)
        self.assertAlmostEqual(F2, F1 / 4, places=10)
    
    def test_zero_distance_error(self):
        """Test that zero distance raises ValueError."""
        with self.assertRaises(ValueError):
            coulomb_force(1e-6, 1e-6, 0)
    
    def test_negative_distance_error(self):
        """Test that negative distance raises ValueError."""
        with self.assertRaises(ValueError):
            coulomb_force(1e-6, 1e-6, -1.0)


class TestElectricField(unittest.TestCase):
    """Test cases for electric field calculations."""
    
    def test_point_charge_field_magnitude(self):
        """Test electric field magnitude from point charge."""
        q = 1e-9  # 1 nC
        r = 0.01  # 1 cm
        
        E = electric_field_point_charge(q, r)
        
        k = constants.value('Coulomb constant')
        expected = k * abs(q) / (r ** 2)
        self.assertAlmostEqual(E, expected, places=2)
    
    def test_field_superposition(self):
        """Test superposition of fields from multiple charges."""
        charges = [1e-9, -1e-9]
        positions = np.array([[1, 0], [-1, 0]])
        field_pos = np.array([0, 1])
        
        E = electric_field_multiple_charges(charges, positions, field_pos)
        
        # At (0, 1), fields should add in y-direction
        self.assertAlmostEqual(E[0], 0, places=10)  # x-component cancels
        self.assertGreater(abs(E[1]), 0)  # y-component non-zero
    
    def test_field_at_charge_location(self):
        """Test field calculation near charge location."""
        charges = [1e-9]
        positions = np.array([[0, 0]])
        field_pos = np.array([0, 0])
        
        # Should handle singularity gracefully
        E = electric_field_multiple_charges(charges, positions, field_pos)
        self.assertTrue(np.allclose(E, 0))


class TestElectricPotential(unittest.TestCase):
    """Test cases for electric potential calculations."""
    
    def test_potential_point_charge(self):
        """Test potential from point charge."""
        q = 1e-9  # 1 nC
        r = 0.01  # 1 cm
        
        V = electric_potential(q, r)
        
        k = constants.value('Coulomb constant')
        expected = k * q / r
        self.assertAlmostEqual(V, expected, places=2)
    
    def test_potential_sign(self):
        """Test potential sign matches charge sign."""
        r = 1.0
        V_pos = electric_potential(1e-9, r)
        V_neg = electric_potential(-1e-9, r)
        
        self.assertGreater(V_pos, 0)
        self.assertLess(V_neg, 0)


class TestMagneticField(unittest.TestCase):
    """Test cases for magnetic field calculations."""
    
    def test_magnetic_field_wire(self):
        """Test magnetic field around current-carrying wire."""
        I = 10.0   # 10 A
        r = 0.01   # 1 cm
        
        B = magnetic_field_wire(I, r)
        
        mu_0 = constants.mu_0
        expected = (mu_0 * I) / (2 * np.pi * r)
        self.assertAlmostEqual(B, expected, places=10)
    
    def test_field_decreases_with_distance(self):
        """Test that B-field decreases with distance."""
        I = 10.0
        B1 = magnetic_field_wire(I, 0.01)
        B2 = magnetic_field_wire(I, 0.02)
        
        self.assertGreater(B1, B2)


class TestWaveProperties(unittest.TestCase):
    """Test cases for electromagnetic wave properties."""
    
    def test_wavelength_from_frequency(self):
        """Test wavelength calculation from frequency."""
        f = 2.4e9  # 2.4 GHz (WiFi)
        wavelength = wavelength_from_frequency(f)
        
        c = constants.c
        expected = c / f
        self.assertAlmostEqual(wavelength, expected, places=5)
        self.assertAlmostEqual(wavelength, 0.1249, places=4)
    
    def test_frequency_from_wavelength(self):
        """Test frequency calculation from wavelength."""
        wavelength = 0.5  # 500 mm
        frequency = frequency_from_wavelength(wavelength)
        
        c = constants.c
        expected = c / wavelength
        self.assertAlmostEqual(frequency, expected, places=2)
    
    def test_reciprocal_relationship(self):
        """Test f = c/λ and λ = c/f are reciprocal."""
        f_original = 1e9
        wavelength = wavelength_from_frequency(f_original)
        f_reconstructed = frequency_from_wavelength(wavelength)
        
        self.assertAlmostEqual(f_original, f_reconstructed, places=2)


class TestFreeSpacePathLoss(unittest.TestCase):
    """Test cases for path loss calculations."""
    
    def test_path_loss_increases_with_distance(self):
        """Test that path loss increases with distance."""
        f = 2.4e9
        loss_10m = free_space_path_loss(10, f)
        loss_100m = free_space_path_loss(100, f)
        
        self.assertGreater(loss_100m, loss_10m)
    
    def test_path_loss_increases_with_frequency(self):
        """Test that path loss increases with frequency."""
        d = 100
        loss_1GHz = free_space_path_loss(d, 1e9)
        loss_10GHz = free_space_path_loss(d, 10e9)
        
        self.assertGreater(loss_10GHz, loss_1GHz)
    
    def test_path_loss_known_value(self):
        """Test path loss against known value."""
        # At 100m and 2.4 GHz, FSPL ≈ 80 dB
        loss = free_space_path_loss(100, 2.4e9)
        self.assertAlmostEqual(loss, 80.05, places=1)


class TestRadiationPattern(unittest.TestCase):
    """Test cases for antenna radiation patterns."""
    
    def test_dipole_pattern_maximum(self):
        """Test that dipole pattern has maximum at θ = π/2."""
        theta = np.linspace(0, np.pi, 181)
        pattern = dipole_radiation_pattern(theta)
        
        # Maximum should be at θ = π/2 (90 degrees)
        max_idx = np.argmax(pattern)
        self.assertAlmostEqual(theta[max_idx], np.pi/2, places=2)
    
    def test_dipole_pattern_normalized(self):
        """Test that pattern is normalized to 1."""
        theta = np.linspace(0, np.pi, 181)
        pattern = dipole_radiation_pattern(theta)
        
        self.assertAlmostEqual(np.max(pattern), 1.0, places=10)
    
    def test_dipole_pattern_nulls(self):
        """Test that dipole has nulls at θ = 0 and θ = π."""
        pattern_0 = dipole_radiation_pattern(0)
        pattern_pi = dipole_radiation_pattern(np.pi)
        
        self.assertAlmostEqual(pattern_0, 0, places=10)
        self.assertAlmostEqual(pattern_pi, 0, places=10)


if __name__ == '__main__':
    unittest.main()
