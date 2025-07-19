# CASTA - Computational Analysis of Apatial Transient Arrests

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
casta.calculate_spatial_transient_wrapper(
    "/path/to/data",
    min_track_length=25,
    dt=0.05,
    plotting_flag=1,
    image_saving_flag="svg"
)
```

## Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `dir` | str | *required* | Path to folder containing track data |
| `--dt` | float | 0.05 | Time step for analysis |
| `--min-track-length` | int | 25 | Minimum track length for analysis |
| `--plot` | int | 0 | Enable plotting (0=off, 1=on) |
| `--image-format` | str | svg | Image format (svg, tiff, png, pdf) |
| `--save_plot` | int | 0 | Enable high-quality image output |

## Input Data Format

CASTA expects track data in CSV format with appropriate trajectory information. The package will process all compatible files in the specified directory.

## Output

The analysis generates:
- **Excel files** with detailed results (`*_CASTA_results.xlsx`)
- **Visualization plots** (optional, in specified format)
- **Console output** with analysis progress and summary

## Requirements

- Python 3.10.18
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn
- hmm-learn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use CASTA in your research, please cite:

```
[citation]
```

## Support

For questions and support, please open an issue on the [GitHub repository](https://github.com/NanoSignalingLab/photochromic-reversion).
