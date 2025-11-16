# Interactive Electromagnetic Fields Learning Environment

An interactive, hands-on learning environment for understanding electromagnetic fields and waves. This project provides Jupyter notebooks with visualizations, calculations, and interactive widgets to explore fundamental concepts in electromagnetism.

## 📚 Course Modules

This learning environment consists of 12 comprehensive modules:

1. **Module 1: Fundamentals of Electric Charge and Fields**
   - Electric charge and conservation
   - Coulomb's Law
   - Electric fields and field lines
   - Interactive force calculators

2. **Module 2: Electric Potential and Capacitance**
   - Electric potential energy
   - Potential difference and voltage
   - Capacitors and dielectrics

3. **Module 3: Current and Resistance**
   - Electric current
   - Ohm's Law and resistance
   - Power and energy in circuits

4. **Module 4: Magnetic Fields and Forces**
   - Magnetic fields from currents
   - Lorentz force
   - Magnetic materials

5. **Module 5: Electromagnetic Induction**
   - Faraday's Law
   - Lenz's Law
   - Inductors and transformers

6. **Module 6: Maxwell's Equations**
   - Gauss's Law
   - Ampère's Law
   - Complete Maxwell's equations

7. **Module 7: Electromagnetic Waves**
   - Wave propagation
   - Energy and momentum in EM waves
   - Poynting vector

8. **Module 8: Reflection and Transmission**
   - Boundary conditions
   - Reflection coefficients
   - Transmission lines

9. **Module 9: Waveguides and Resonators**
   - Rectangular and circular waveguides
   - Cavity resonators
   - Quality factor

10. **Module 10: Antennas and Radiation**
    - Dipole antennas
    - Radiation patterns
    - Antenna parameters

11. **Module 11: Wireless Propagation**
    - Path loss models
    - Multipath effects
    - Link budget analysis

12. **Module 12: Advanced Topics**
    - Metamaterials
    - Photonics
    - Modern applications

## 🚀 Installation

### Using pip (Recommended)

1. Clone this repository:
```bash
git clone https://github.com/mohbagher/untitled.git
cd untitled
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Using Conda

1. Clone this repository:
```bash
git clone https://github.com/mohbagher/untitled.git
cd untitled
```

2. Create the conda environment:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate em-fields
```

## 📖 Getting Started

1. **Launch Jupyter:**
```bash
jupyter notebook
```
or
```bash
jupyter lab
```

2. **Start with the introduction:**
   - Open `notebooks/00_getting_started.ipynb` for a quick introduction
   - Then proceed to `notebooks/01_fundamentals.ipynb` for Module 1

3. **Follow the learning path:**
   - Work through modules sequentially
   - Complete interactive exercises
   - Experiment with the visualization tools

## 📁 Repository Structure

```
untitled/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies (pip)
├── environment.yml                    # Conda environment specification
├── .gitignore                        # Git ignore file
├── src/                              # Source code
│   ├── __init__.py                   # Package initialization
│   ├── calculations.py               # Electromagnetic calculations
│   ├── visualizations.py             # Plotting and visualization functions
│   ├── interactive_widgets.py        # Interactive widget utilities
│   └── animations.py                 # Animation generators
├── data/                             # Data files
│   └── material_properties.json      # Material properties database
├── notebooks/                        # Jupyter notebooks
│   ├── 00_getting_started.ipynb      # Introduction and setup
│   ├── 01_fundamentals.ipynb         # Module 1: Charge and fields
│   ├── 02_potential.ipynb            # Module 2: Potential (coming soon)
│   └── ...                           # Additional modules
└── tests/                            # Unit tests
    └── test_calculations.py          # Tests for calculation functions
```

## 🎓 Learning Path

### For Beginners
1. Start with Module 1 to understand basic concepts
2. Use interactive widgets to build intuition
3. Complete quiz questions to test understanding
4. Experiment with parameters in visualizations

### For Advanced Learners
- Jump to specific modules of interest
- Modify code examples for custom scenarios
- Explore the source code in `src/` directory
- Extend visualizations with your own analysis

## 🛠️ Troubleshooting

### Jupyter Notebook Issues

**Problem:** Widgets not displaying
```bash
# Enable ipywidgets extension
jupyter nbextension enable --py widgetsnbextension
```

**Problem:** Interactive plots not working
```bash
# For JupyterLab, ensure extensions are installed
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyterlab-plotly
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Add the project root to Python path in your notebook:
```python
import sys
sys.path.append('..')  # If running from notebooks/ directory
```

Or install the package in development mode:
```bash
pip install -e .
```

### Visualization Issues

**Problem:** Plots not rendering properly

**Solution:** Try different backends:
```python
# For notebook
%matplotlib inline

# For interactive plots
%matplotlib widget
```

### Performance Issues

**Problem:** Slow visualizations with many data points

**Solution:** Reduce the resolution or number of points:
```python
# In visualization functions, reduce grid size
x = np.linspace(-5, 5, 20)  # Instead of 50 or 100
```

## 💡 Tips for Best Experience

- Use JupyterLab for the best interactive experience
- Enable "Trust" on notebooks to see all interactive widgets
- Restart kernel if widgets stop responding
- Save your work frequently
- Experiment with code - it's the best way to learn!

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Add more example problems
- Improve documentation
- Submit pull requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with NumPy, SciPy, Matplotlib, and Plotly
- Interactive widgets powered by ipywidgets
- Educational content inspired by classical electromagnetism textbooks

---

**Happy Learning!** 🎓⚡📊
