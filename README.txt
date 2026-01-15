This archive contains the numerical scripts used in the analysis reported in the associated manuscript.

The code implements the deterministic numerical pipeline described in the paper, including:
- the ultraviolet (UV) analysis based on Higgs vacuum stability,
- the infrared (IR) analysis based on cosmological linear response,
- and the supplementary robustness tests.

The scripts are provided in the exact form used for the production of the results reported in the manuscript.
No tuning or post-processing is performed within the archive.

Reference numerical outputs, when provided, are intended for verification purposes only.
The documented software environment allows re-execution on compliant platforms.

## Reproducibility (Google Colab)

This analysis was executed with:
- Python 3.12.12
- NumPy 2.0.2
- SciPy 1.16.3
- Matplotlib 3.10.0

To reproduce in Colab:
1. Run:
   pip install -r https://raw.githubusercontent.com/moranberraho/asymptotic_safety_consistency_test/main/requirements.txt
2. Restart the runtime
3. Run `main_analysis.py` and `robustness_analysis.py`

