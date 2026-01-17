This archive contains the numerical scripts used in the analysis reported in the associated manuscript.

The code implements the deterministic numerical pipeline described in the paper, including:
- the ultraviolet (UV) analysis based on Higgs vacuum stability,
- the infrared (IR) analysis based on cosmological linear response,
- a dedicated kernel validation module: This script quantifies the response of the kernel to realistic variations of the observational window geometry and background cosmological parameters.
- and the supplementary robustness tests.

The scripts are provided in the exact form used for the production of the results reported in the manuscript.
No tuning or post-processing is performed within the archive.
The kernel sensitivity analysis does not introduce new physical assumptions or additional model parameters. It is used exclusively as a robustness and validation layer supporting the infrared uncertainty budget discussed in Appendix D.6 of the manuscript.


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
3. Clone the repository and move into the project directory:
   git clone https://github.com/moranberraho/asymptotic_safety_consistency_test.git
   cd asymptotic_safety_consistency_test

4. (Optional but recommended) Verify the execution environment:
   Run the following in a notebook cell to confirm that the pinned versions are used:
   
   import sys, numpy, scipy, matplotlib
   print("Python:", sys.version)
   print("NumPy:", numpy.__version__)
   print("SciPy:", scipy.__version__)
   print("Matplotlib:", matplotlib.__version__)

   Expected environment:
   - Python 3.12.12
   - NumPy 2.0.2
   - SciPy 1.16.3
   - Matplotlib 3.10.0

5. Run the main analysis:
   python main_analysis.py

   This will execute the full UV–IR consistency test and print:
   - the UV inference from Higgs vacuum stability,
   - the IR inference from gravitational slip,
   - the combined statistical tension metric T,
   together with a structured summary of inputs and results.

6. Run the robustness analysis:
	a) Main robustness tests:
      python robustness_analysis.py

   	b) Infrared slip robustness scan:
      python slip_analysis.py 

       c) Infrared kernel sensitivity validation:
       python kernel_analysis.py  

   This will perform:
   - activation profile robustness tests,
   - threshold factor F sensitivity scans,
   - supplementary consistency checks,
   - a systematic scan of the gravitational slip parameter and quantifies the stability of the UV–IR consistency relation.
   and will generate the figures used in the manuscript’s supplementary material.
   - a controlled scan of the infrared transfer kernel Under variations of the observational window and background cosmological parameters, and reproduces the numerical values summarized in Appendix D.6 of the manuscript.

7. Verify reproducibility:
   The numerical outputs (central values, uncertainties, and tension metric)
   should match those reported in the manuscript (Table A.1 and Sections 5–6),
   up to machine precision.

   Successful completion is indicated by:
   - "Analysis completed successfully!"
   - "Both robustness tests PASSED"

The slip parameter robustness analysis is conditionally dependent on the ultraviolet
constraint obtained from the main pipeline; it is therefore intended as a consistency
and stability test, not as an independent UV inference.

No external datasets are required. All inputs are self-contained, and the
analysis is fully deterministic.

All scripts in this repository are archived under the same Zenodo concept DOI.
Individual releases correspond to versioned snapshots of the full analysis pipeline.





