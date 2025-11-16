"""
Electromagnetic Field Calculations Module

This module provides functions for calculating various electromagnetic quantities
including electric fields, magnetic fields, forces, and wave properties.
"""

import numpy as np
from scipy import constants


def coulomb_force(q1, q2, r):
    """
    Calculate the electrostatic force between two point charges using Coulomb's Law.
    
    F = k * |q1 * q2| / r^2
    
    Parameters
    ----------
    q1 : float
        First charge in Coulombs (C)
    q2 : float
        Second charge in Coulombs (C)
    r : float
        Distance between charges in meters (m)
    
    Returns
    -------
    tuple
        (force_magnitude, force_type) where:
        - force_magnitude: float, magnitude of force in Newtons (N)
        - force_type: str, either 'attractive' or 'repulsive'
    
    Raises
    ------
    ValueError
        If distance r is zero or negative
    
    Examples
    --------
    >>> F, ftype = coulomb_force(1e-6, -2e-6, 0.1)
    >>> print(f"Force: {F:.4f} N, Type: {ftype}")
    Force: 1.7975 N, Type: attractive
    """
    if r <= 0:
        raise ValueError("Distance must be positive and non-zero")
    
    # Coulomb's constant k = 1/(4πε₀)
    k = 1 / (4 * np.pi * constants.epsilon_0)  # 8.987551787e9 N⋅m²/C²
    
    # Calculate force magnitude
    force_magnitude = k * abs(q1 * q2) / (r ** 2)
    
    # Determine force type based on charge signs
    if (q1 * q2) > 0:
        force_type = 'repulsive'
    else:
        force_type = 'attractive'
    
    return force_magnitude, force_type


def electric_field_point_charge(q, r, position=None):
    """
    Calculate the electric field at a point due to a point charge.
    
    E = k * q / r^2 * r_hat
    
    Parameters
    ----------
    q : float
        Charge in Coulombs (C)
    r : float or numpy.ndarray
        Distance(s) from the charge in meters (m)
    position : numpy.ndarray, optional
        3D position vector(s) where field is calculated.
        If provided, returns vector field; otherwise returns magnitude only.
    
    Returns
    -------
    float or numpy.ndarray
        Electric field magnitude in N/C (or vector field if position provided)
    
    Examples
    --------
    >>> E = electric_field_point_charge(1e-9, 0.01)
    >>> print(f"E-field magnitude: {E:.4f} N/C")
    E-field magnitude: 89875.5179 N/C
    """
    k = 1 / (4 * np.pi * constants.epsilon_0)
    
    if position is not None:
        # Calculate vector field
        r_vec = np.asarray(position)
        r_mag = np.linalg.norm(r_vec)
        if r_mag == 0:
            raise ValueError("Position cannot be at charge location")
        r_hat = r_vec / r_mag
        E_vec = k * q / (r_mag ** 2) * r_hat
        return E_vec
    else:
        # Calculate magnitude only
        if np.any(r <= 0):
            raise ValueError("Distance must be positive and non-zero")
        E_magnitude = k * abs(q) / (r ** 2)
        return E_magnitude


def electric_field_multiple_charges(charges, charge_positions, field_position):
    """
    Calculate the electric field at a point due to multiple point charges
    using the principle of superposition.
    
    Parameters
    ----------
    charges : list or numpy.ndarray
        Array of charges in Coulombs (C)
    charge_positions : numpy.ndarray
        Array of charge positions, shape (N, 2) or (N, 3)
    field_position : numpy.ndarray
        Position where field is calculated, shape (2,) or (3,)
    
    Returns
    -------
    numpy.ndarray
        Electric field vector at the specified position in N/C
    
    Examples
    --------
    >>> charges = [1e-9, -1e-9]
    >>> positions = np.array([[0, 0], [1, 0]])
    >>> field_pos = np.array([0.5, 0.5])
    >>> E = electric_field_multiple_charges(charges, positions, field_pos)
    """
    k = 1 / (4 * np.pi * constants.epsilon_0)
    charges = np.asarray(charges)
    charge_positions = np.asarray(charge_positions)
    field_position = np.asarray(field_position)
    
    # Initialize total field
    E_total = np.zeros_like(field_position, dtype=float)
    
    # Sum contributions from each charge
    for q, pos in zip(charges, charge_positions):
        r_vec = field_position - pos
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag > 1e-10:  # Avoid singularity at charge location
            r_hat = r_vec / r_mag
            E_contribution = k * q / (r_mag ** 2) * r_hat
            E_total += E_contribution
    
    return E_total


def electric_potential(q, r):
    """
    Calculate the electric potential (voltage) at a distance from a point charge.
    
    V = k * q / r
    
    Parameters
    ----------
    q : float
        Charge in Coulombs (C)
    r : float or numpy.ndarray
        Distance(s) from the charge in meters (m)
    
    Returns
    -------
    float or numpy.ndarray
        Electric potential in Volts (V)
    
    Examples
    --------
    >>> V = electric_potential(1e-9, 0.01)
    >>> print(f"Potential: {V:.4f} V")
    Potential: 898.7552 V
    """
    if np.any(r <= 0):
        raise ValueError("Distance must be positive and non-zero")
    
    k = 1 / (4 * np.pi * constants.epsilon_0)
    V = k * q / r
    return V


def magnetic_field_wire(I, r):
    """
    Calculate the magnetic field around a long straight current-carrying wire.
    
    B = (μ₀ * I) / (2π * r)
    
    Parameters
    ----------
    I : float
        Current in Amperes (A)
    r : float or numpy.ndarray
        Distance(s) from the wire in meters (m)
    
    Returns
    -------
    float or numpy.ndarray
        Magnetic field magnitude in Tesla (T)
    
    Examples
    --------
    >>> B = magnetic_field_wire(10, 0.01)
    >>> print(f"B-field: {B:.6e} T")
    B-field: 2.000000e-04 T
    """
    if np.any(r <= 0):
        raise ValueError("Distance must be positive and non-zero")
    
    mu_0 = constants.mu_0  # Permeability of free space
    B = (mu_0 * I) / (2 * np.pi * r)
    return B


def wavelength_from_frequency(frequency):
    """
    Calculate wavelength from frequency for electromagnetic waves.
    
    λ = c / f
    
    Parameters
    ----------
    frequency : float
        Frequency in Hertz (Hz)
    
    Returns
    -------
    float
        Wavelength in meters (m)
    
    Examples
    --------
    >>> wavelength = wavelength_from_frequency(2.4e9)  # WiFi frequency
    >>> print(f"Wavelength: {wavelength:.4f} m")
    Wavelength: 0.1249 m
    """
    if frequency <= 0:
        raise ValueError("Frequency must be positive")
    
    c = constants.c  # Speed of light
    wavelength = c / frequency
    return wavelength


def frequency_from_wavelength(wavelength):
    """
    Calculate frequency from wavelength for electromagnetic waves.
    
    f = c / λ
    
    Parameters
    ----------
    wavelength : float
        Wavelength in meters (m)
    
    Returns
    -------
    float
        Frequency in Hertz (Hz)
    
    Examples
    --------
    >>> freq = frequency_from_wavelength(0.5)  # 500 mm
    >>> print(f"Frequency: {freq:.2e} Hz")
    Frequency: 5.99e+08 Hz
    """
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive")
    
    c = constants.c  # Speed of light
    frequency = c / wavelength
    return frequency


def free_space_path_loss(distance, frequency):
    """
    Calculate free space path loss (FSPL) for wireless signal propagation.
    
    FSPL(dB) = 20*log₁₀(d) + 20*log₁₀(f) + 20*log₁₀(4π/c)
    
    Parameters
    ----------
    distance : float
        Distance between transmitter and receiver in meters (m)
    frequency : float
        Signal frequency in Hertz (Hz)
    
    Returns
    -------
    float
        Path loss in decibels (dB)
    
    Examples
    --------
    >>> loss = free_space_path_loss(100, 2.4e9)
    >>> print(f"Path loss at 100m: {loss:.2f} dB")
    Path loss at 100m: 80.05 dB
    """
    if distance <= 0 or frequency <= 0:
        raise ValueError("Distance and frequency must be positive")
    
    c = constants.c
    # FSPL in dB
    fspl_db = 20 * np.log10(distance) + 20 * np.log10(frequency) + \
              20 * np.log10(4 * np.pi / c)
    
    return fspl_db


def dipole_radiation_pattern(theta, I0=1.0, length=None, wavelength=1.0):
    """
    Calculate the radiation pattern of a dipole antenna.
    
    For a short dipole (Hertzian dipole):
    E(θ) ∝ sin(θ)
    
    Parameters
    ----------
    theta : float or numpy.ndarray
        Angle(s) from the dipole axis in radians
    I0 : float, optional
        Current amplitude (default: 1.0)
    length : float, optional
        Dipole length in meters (if None, assumes short dipole)
    wavelength : float, optional
        Wavelength in meters (default: 1.0)
    
    Returns
    -------
    numpy.ndarray
        Normalized radiation pattern values
    
    Examples
    --------
    >>> theta = np.linspace(0, np.pi, 100)
    >>> pattern = dipole_radiation_pattern(theta)
    >>> # Maximum at θ = π/2 (perpendicular to dipole)
    """
    theta = np.asarray(theta)
    
    # Hertzian (short) dipole pattern
    if length is None or length < wavelength / 10:
        pattern = np.abs(np.sin(theta))
        # Handle numerical precision issues at 0 and π
        pattern = np.where(np.abs(pattern) < 1e-10, 0, pattern)
    else:
        # Half-wave dipole approximation
        pattern = np.cos(np.pi / 2 * np.cos(theta)) / np.sin(theta)
        # Handle singularities at theta=0 and theta=π
        pattern = np.where(np.abs(np.sin(theta)) < 1e-10, 0, pattern)
        pattern = np.abs(pattern)
    
    # Normalize
    max_val = np.max(pattern)
    if max_val > 0:
        pattern = pattern / max_val
    
    return pattern
