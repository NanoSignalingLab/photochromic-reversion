# CASTA - Computational Analysis of Spatial Transient Arrests

CASTA is a Python package for analyzing spatial transient patterns in tracking data using Hidden Markov Models (HMM). It provides tools for processing and plotting trajectory data.

## Installation

### From PyPI (recommended)

```bash
pip install casta
```

### For development

```bash
git clone https://github.com/NanoSignalingLab/photochromic-reversion.git
cd photochromic-reversion
pip install -e .
```

## Quick Start

### Command Line Usage

```bash
# Basic usage
python -m casta /path/to/your/track/data

# With parameters
python -m casta /path/to/data --dt 0.05 --min-track-length 25
```

### Jupyter Notebook Usage

```python
import casta

# Run analysis in notebook
casta.calculate_sta(
    dir="/path/to/data/directory",
    out_dir="/path/to/output/directory",
    min_track_length=25,
    dt=0.05,
    plot=True,
    image_format="svg"
)
```

## Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `dir` | str | *required* | Path to directory containing input track data |
| `--out_dir` | str | None | Path to output directory to save results, defaults to input directory |
| `--dt` | float | 0.05 | Time step for analysis |
| `--min-track-length` | int | 25 | Minimum track length for analysis |
| `--plot` | bool | False | Enable additional plotting |
| `--image-format` | str | svg | Image format (svg, tiff) |

## Input Data Format

CASTA includes an example file.

```python
import os
import casta

example_df, path = casta.utils.load_example_data()

current_dir = os.getcwd()

casta.calculate_sta(path, out_dir=current_dir)
```

## Output

The analysis generates:
- **Excel files** with detailed results (`*_CASTA_results.xlsx`)
- **Visualization plots** (optional, in specified format)

## Requirements

- Python 3.10.18
- NumPy 1.26.4
- Pandas 2.2.3
- Matplotlib 3.10.0
- SciPy 1.15.0
- Scikit-learn 1.6.1
- Seaborn 0.13.2
- hmm-learn 0.3.3
- Shapely 2.0.6
- xlsxwriter 3.2.3

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use CASTA in your research, please cite:

```
Photochromic reversion enables long-term tracking of single molecules in living plants
Michelle von Arx, Kaltra Xhelilaj, Philip Schulz, Sven zur Oven-Krockhaus, Julien Gronnier
bioRxiv 2024.04.10.585335; doi: https://doi.org/10.1101/2024.04.10.585335
```

## Support

For questions and support, please open an issue on the [GitHub repository](https://github.com/NanoSignalingLab/photochromic-reversion).
