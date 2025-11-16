"""
Visualization Functions for Electromagnetic Fields

This module provides plotting functions for visualizing electric and magnetic fields,
radiation patterns, and other electromagnetic phenomena.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import plotly.graph_objects as go
from scipy import constants

from .calculations import electric_field_multiple_charges


class Arrow3D:
    """
    Helper class for drawing 3D arrows in matplotlib.
    
    This class creates arrow objects that can be added to 3D matplotlib plots.
    
    Attributes
    ----------
    arrow : FancyArrowPatch
        The matplotlib arrow patch object
    
    Examples
    --------
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> arrow = Arrow3D([0, 1], [0, 1], [0, 1], color='red')
    >>> ax.add_artist(arrow.arrow)
    """
    
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """
        Initialize a 3D arrow.
        
        Parameters
        ----------
        xs : list
            X coordinates [start, end]
        ys : list
            Y coordinates [start, end]
        zs : list
            Z coordinates [start, end]
        *args, **kwargs
            Additional arguments passed to FancyArrowPatch
        """
        from matplotlib.patches import FancyArrowPatch
        from mpl_toolkits.mplot3d import proj3d
        
        class Arrow3DPatch(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
                self._verts3d = xs, ys, zs
            
            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                FancyArrowPatch.draw(self, renderer)
        
        self.arrow = Arrow3DPatch(xs, ys, zs, *args, **kwargs)


def plot_electric_field_2d(charges, charge_positions, xlim=(-5, 5), ylim=(-5, 5),
                           grid_points=20, plot_type='both', ax=None, title=None):
    """
    Plot 2D electric field vectors and/or streamlines for multiple charges.
    
    Parameters
    ----------
    charges : list or numpy.ndarray
        Array of charges in Coulombs (C)
    charge_positions : numpy.ndarray
        Array of charge positions, shape (N, 2)
    xlim : tuple, optional
        X-axis limits (default: (-5, 5))
    ylim : tuple, optional
        Y-axis limits (default: (-5, 5))
    grid_points : int, optional
        Number of grid points in each dimension (default: 20)
    plot_type : str, optional
        Type of plot: 'vectors', 'streamlines', or 'both' (default: 'both')
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new if None)
    title : str, optional
        Plot title
    
    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    
    Examples
    --------
    >>> charges = [1e-9, -1e-9]
    >>> positions = np.array([[1, 0], [-1, 0]])
    >>> fig, ax = plot_electric_field_2d(charges, positions)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    charges = np.asarray(charges)
    charge_positions = np.asarray(charge_positions)
    
    # Create grid
    x = np.linspace(xlim[0], xlim[1], grid_points)
    y = np.linspace(ylim[0], ylim[1], grid_points)
    X, Y = np.meshgrid(x, y)
    
    # Calculate electric field at each grid point
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            field_pos = np.array([X[i, j], Y[i, j]])
            E = electric_field_multiple_charges(charges, charge_positions, field_pos)
            Ex[i, j] = E[0]
            Ey[i, j] = E[1]
    
    # Calculate field magnitude for coloring
    E_mag = np.sqrt(Ex**2 + Ey**2)
    
    # Plot based on type
    if plot_type in ['vectors', 'both']:
        # Normalize vectors for better visualization
        E_mag_plot = np.log10(E_mag + 1e-10)  # Log scale for better visibility
        ax.quiver(X, Y, Ex, Ey, E_mag_plot, cmap='plasma', alpha=0.7,
                 scale=None, scale_units='xy')
    
    if plot_type in ['streamlines', 'both']:
        # Plot streamlines
        ax.streamplot(X, Y, Ex, Ey, color=E_mag, cmap='viridis',
                     linewidth=1.5, density=1.5, arrowsize=1.5)
    
    # Plot charge positions
    for q, pos in zip(charges, charge_positions):
        if q > 0:
            color = 'red'
            marker = '+'
            label = f'+{abs(q)*1e9:.1f} nC'
        else:
            color = 'blue'
            marker = '_'
            label = f'-{abs(q)*1e9:.1f} nC'
        
        # Scale circle size with charge magnitude
        circle_size = 0.3 * np.log10(abs(q) * 1e9 + 1)
        circle = Circle(pos, circle_size, color=color, alpha=0.5, zorder=10)
        ax.add_patch(circle)
        ax.plot(pos[0], pos[1], marker, markersize=20, color='white',
               markeredgewidth=3, zorder=11)
        ax.text(pos[0], pos[1] - 0.7, label, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Electric Field Visualization', fontsize=14, fontweight='bold')
    
    return fig, ax


def plot_electric_field_3d_plotly(charges, charge_positions, xlim=(-5, 5),
                                   ylim=(-5, 5), zlim=(-5, 5), grid_points=10):
    """
    Create an interactive 3D visualization of electric field using Plotly.
    
    Parameters
    ----------
    charges : list or numpy.ndarray
        Array of charges in Coulombs (C)
    charge_positions : numpy.ndarray
        Array of charge positions, shape (N, 3)
    xlim : tuple, optional
        X-axis limits (default: (-5, 5))
    ylim : tuple, optional
        Y-axis limits (default: (-5, 5))
    zlim : tuple, optional
        Z-axis limits (default: (-5, 5))
    grid_points : int, optional
        Number of grid points in each dimension (default: 10)
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D plotly figure
    
    Examples
    --------
    >>> charges = [1e-9, -1e-9]
    >>> positions = np.array([[1, 0, 0], [-1, 0, 0]])
    >>> fig = plot_electric_field_3d_plotly(charges, positions)
    >>> fig.show()
    """
    charges = np.asarray(charges)
    charge_positions = np.asarray(charge_positions)
    
    # Create 3D grid
    x = np.linspace(xlim[0], xlim[1], grid_points)
    y = np.linspace(ylim[0], ylim[1], grid_points)
    z = np.linspace(zlim[0], zlim[1], grid_points)
    
    # Sample points (not full meshgrid to reduce data)
    points = []
    vectors = []
    
    for xi in x:
        for yi in y:
            for zi in z:
                field_pos = np.array([xi, yi, zi])
                E = electric_field_multiple_charges(charges, charge_positions, field_pos)
                E_mag = np.linalg.norm(E)
                
                if E_mag > 1e-10:  # Only plot significant fields
                    points.append(field_pos)
                    # Normalize vector for visualization
                    vectors.append(E / E_mag * 0.5)
    
    points = np.array(points)
    vectors = np.array(vectors)
    
    # Create figure
    fig = go.Figure()
    
    # Add charge positions
    for q, pos in zip(charges, charge_positions):
        color = 'red' if q > 0 else 'blue'
        size = 10 * np.log10(abs(q) * 1e9 + 1)
        
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            marker=dict(size=size, color=color, opacity=0.8),
            name=f"{'+'if q>0 else ''}{q*1e9:.1f} nC",
            showlegend=True
        ))
    
    # Add field vectors as cones
    if len(points) > 0:
        fig.add_trace(go.Cone(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            u=vectors[:, 0],
            v=vectors[:, 1],
            w=vectors[:, 2],
            colorscale='Viridis',
            sizemode="absolute",
            sizeref=0.5,
            showscale=True,
            name='E-field'
        ))
    
    # Update layout
    fig.update_layout(
        title='Interactive 3D Electric Field',
        scene=dict(
            xaxis_title='x (m)',
            yaxis_title='y (m)',
            zaxis_title='z (m)',
            xaxis=dict(range=xlim),
            yaxis=dict(range=ylim),
            zaxis=dict(range=zlim),
            aspectmode='cube'
        ),
        width=800,
        height=700
    )
    
    return fig


def plot_coulomb_force_vs_distance(q1, q2, r_min=0.01, r_max=1.0, n_points=100,
                                   ax=None, log_scale=True):
    """
    Plot Coulomb force magnitude as a function of distance.
    
    Parameters
    ----------
    q1 : float
        First charge in Coulombs (C)
    q2 : float
        Second charge in Coulombs (C)
    r_min : float, optional
        Minimum distance in meters (default: 0.01)
    r_max : float, optional
        Maximum distance in meters (default: 1.0)
    n_points : int, optional
        Number of points to plot (default: 100)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new if None)
    log_scale : bool, optional
        Use logarithmic scale for both axes (default: True)
    
    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    
    Examples
    --------
    >>> fig, ax = plot_coulomb_force_vs_distance(1e-6, -2e-6)
    >>> plt.show()
    """
    from .calculations import coulomb_force
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Generate distance array
    if log_scale:
        r = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
    else:
        r = np.linspace(r_min, r_max, n_points)
    
    # Calculate forces
    forces = []
    for ri in r:
        F, _ = coulomb_force(q1, q2, ri)
        forces.append(F)
    
    forces = np.array(forces)
    
    # Determine force type
    _, force_type = coulomb_force(q1, q2, r_min)
    color = 'red' if force_type == 'repulsive' else 'blue'
    
    # Plot
    ax.plot(r, forces, linewidth=2.5, color=color, label=f'{force_type.capitalize()} force')
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Force Magnitude (N)', fontsize=12)
    ax.set_title(f'Coulomb Force: q₁={q1*1e6:.2f}μC, q₂={q2*1e6:.2f}μC',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11)
    
    # Add 1/r² reference line
    if log_scale:
        ref_r = r
        ref_F = forces[0] * (r[0] / ref_r) ** 2
        ax.plot(ref_r, ref_F, '--', color='gray', alpha=0.5, label='1/r² reference')
        ax.legend(fontsize=11)
    
    return fig, ax


def plot_radiation_pattern_2d(theta, pattern, ax=None, polar=True, title=None):
    """
    Plot antenna radiation pattern in 2D.
    
    Parameters
    ----------
    theta : numpy.ndarray
        Angles in radians
    pattern : numpy.ndarray
        Normalized radiation pattern values
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new if None)
    polar : bool, optional
        Use polar plot (default: True)
    title : str, optional
        Plot title
    
    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    
    Examples
    --------
    >>> theta = np.linspace(0, 2*np.pi, 360)
    >>> pattern = dipole_radiation_pattern(theta)
    >>> fig, ax = plot_radiation_pattern_2d(theta, pattern)
    >>> plt.show()
    """
    if ax is None:
        if polar:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    if polar:
        ax.plot(theta, pattern, linewidth=2.5, color='darkblue')
        ax.fill(theta, pattern, alpha=0.3, color='lightblue')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Normalized Pattern', fontsize=11)
        ax.grid(True, alpha=0.5)
    else:
        ax.plot(np.degrees(theta), pattern, linewidth=2.5, color='darkblue')
        ax.fill_between(np.degrees(theta), pattern, alpha=0.3, color='lightblue')
        ax.set_xlabel('Angle (degrees)', fontsize=12)
        ax.set_ylabel('Normalized Pattern', fontsize=12)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title('Radiation Pattern', fontsize=14, fontweight='bold', pad=20)
    
    return fig, ax
