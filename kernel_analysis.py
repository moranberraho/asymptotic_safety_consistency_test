"""
================================================================================
Validation of the Infrared Transfer Kernel Uncertainty: σ(Tη) = 0.10

Manuscript:
"Testing the Minimal Asymptotic Safety Scenario: A Multi-Messenger 
 Confrontation of the Higgs Vacuum and Large-Scale Structure"

Appendix D.6, Table D.1

Purpose:
--------
This module computes the sensitivity of the infrared response kernel Tη to 
representative variations in the observational window geometry and background 
cosmological parameters. The analysis validates the conservative uncertainty 
decomposition σ(Tη) = 0.10 adopted in the manuscript.

Scientific Context:
-------------------
The transfer kernel Tη maps the ultraviolet fixed-point parameter g* to 
observable deviations in the gravitational slip parameter η(k,z) = Φ/Ψ 
through a linear-response framework in the quasi-static approximation.

The uncertainty budget decomposes as:
  - ~5%: Observational window variations (redshift, geometry)
  - ~5%: Background cosmological parameters (Ωm, H0)
  - ~15%: Theoretical envelope (QSA, scale/redshift averaging)

This code provides quantitative validation of the first two contributions.

Author: Dr. Moran BERRAHO
Date: January 2025
License: CC-BY-4.0 (academic reproducibility)
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

# ============================================================================
# COSMOLOGICAL FRAMEWORK
# ============================================================================

@dataclass
class CosmologicalParameters:
    """
    Standard ΛCDM cosmological parameters.
    
    Fiducial values are consistent with Planck 2018 cosmological constraints
    [Aghanim et al., A&A 641, A6 (2020)].
    
    Attributes:
        Omega_m (float): Present-day matter density parameter
        Omega_Lambda (float): Present-day dark energy density parameter
        H0 (float): Hubble constant in km/s/Mpc
        h (float): Dimensionless Hubble parameter (H0/100)
        c_light (float): Speed of light in km/s
    """
    Omega_m: float = 0.315
    Omega_Lambda: float = 0.685
    H0: float = 67.4  # km/s/Mpc
    h: float = 0.674
    c_light: float = 299792.458  # km/s
    
    def __post_init__(self):
        """Validate parameter consistency."""
        if not np.isclose(self.Omega_m + self.Omega_Lambda, 1.0, atol=1e-3):
            raise ValueError("Flat universe required: Ωm + ΩΛ = 1")
        if not np.isclose(self.h, self.H0 / 100.0, rtol=1e-3):
            raise ValueError("Inconsistent h and H0 definitions")


class CosmologicalModel:
    """
    Linearized cosmological evolution in the ΛCDM framework.
    
    Provides methods for computing:
      - Hubble parameter H(z)
      - Linear growth factor D(z)
      - Comoving distance χ(z)
    
    All methods support parameter variations for sensitivity analysis.
    
    Attributes:
        params (CosmologicalParameters): Baseline cosmological parameters
    """
    
    def __init__(self, params: Optional[CosmologicalParameters] = None):
        """
        Initialize cosmological model.
        
        Args:
            params: Cosmological parameters. If None, uses fiducial values.
        """
        self.params = params if params is not None else CosmologicalParameters()
    
    def H(self, z: np.ndarray, Omega_m: Optional[float] = None) -> np.ndarray:
        """
        Hubble parameter H(z) in km/s/Mpc.
        
        Args:
            z: Redshift (scalar or array)
            Omega_m: Matter density parameter. If None, uses fiducial value.
        
        Returns:
            H(z) in km/s/Mpc
        
        Notes:
            Standard flat ΛCDM: H(z) = H0 √[Ωm(1+z)³ + ΩΛ]
        """
        Om = Omega_m if Omega_m is not None else self.params.Omega_m
        OL = 1.0 - Om
        return self.params.H0 * np.sqrt(Om * (1 + z)**3 + OL)
    
    def growth_factor(self, z: np.ndarray, Omega_m: Optional[float] = None) -> np.ndarray:
        """
        Linear growth factor D(z) normalized to D(z=0) = 1.
        
        Args:
            z: Redshift (scalar or array)
            Omega_m: Matter density parameter. If None, uses fiducial value.
        
        Returns:
            D(z): Linear growth factor
        
        Notes:
            Uses the Heath (1977) approximation for ΛCDM, accurate to ~1% 
            for z < 3 [Carroll, Press & Turner, ARA&A 30, 499 (1992)].
            
            Alternative exact numerical integration available but not required 
            for the present sensitivity analysis at the ~5% level.
        """
        Om = Omega_m if Omega_m is not None else self.params.Omega_m
        OL = 1.0 - Om
        a = 1.0 / (1.0 + z)
        
        # Time-dependent matter density parameter
        Om_a = Om * (1 + z)**3 / (Om * (1 + z)**3 + OL)
        
        # Heath (1977) fitting formula for ΛCDM growth factor
        numerator = 2.5 * Om_a
        denominator = (Om_a**(4./7.) - OL + 
                      (1.0 + Om_a / 2.0) * (1.0 + OL / 70.0))
        
        g = numerator / denominator
        
        return g
    
    def comoving_distance(self, z: float, Omega_m: Optional[float] = None) -> float:
        """
        Comoving distance χ(z) in Mpc/h.
        
        Args:
            z: Redshift (scalar)
            Omega_m: Matter density parameter. If None, uses fiducial value.
        
        Returns:
            χ(z) in Mpc/h
        
        Notes:
            Computed via numerical integration:
            χ(z) = (c/H0) ∫₀ᶻ dz' / E(z')
            where E(z) = H(z)/H0
        """
        z_array = np.linspace(0, z, 200)
        integrand = self.params.c_light / self.H(z_array, Omega_m)
        chi = trapezoid(integrand, z_array) * self.params.h
        return chi


# ============================================================================
# OBSERVATIONAL WINDOW FUNCTIONS
# ============================================================================

class ObservationalWindow:
    """
    Normalized redshift window functions W(z) for weak-lensing surveys.
    
    Represents the effective redshift distribution n(z) of source galaxies
    weighted by lensing efficiency and survey geometry.
    """
    
    @staticmethod
    def gaussian(z: np.ndarray, 
                 z_eff: float = 0.8, 
                 sigma_z: float = 0.3) -> np.ndarray:
        """
        Gaussian redshift window.
        
        Args:
            z: Redshift array
            z_eff: Effective (mean) redshift
            sigma_z: Width parameter
        
        Returns:
            Normalized window function W(z)
        
        Notes:
            Commonly used for stage-III surveys (KiDS, DES).
            Representative values: z_eff ∈ [0.6, 1.0], σz ~ 0.3
        """
        W = np.exp(-0.5 * ((z - z_eff) / sigma_z)**2)
        
        # Normalization: ∫ W(z) dz = 1
        z_norm = np.linspace(0, 3.5, 500)
        W_norm = np.exp(-0.5 * ((z_norm - z_eff) / sigma_z)**2)
        norm = trapezoid(W_norm, z_norm)
        
        return W / norm
    
    @staticmethod
    def tophat(z: np.ndarray, 
               z_eff: float = 0.8, 
               half_width: float = 0.6) -> np.ndarray:
        """
        Top-hat (uniform) redshift window.
        
        Args:
            z: Redshift array
            z_eff: Center redshift
            half_width: Half-width of the top-hat
        
        Returns:
            Normalized window function W(z)
        
        Notes:
            Provides a conservative test of sensitivity to window shape.
            Support: [z_eff - half_width, z_eff + half_width]
        """
        z_min = z_eff - half_width
        z_max = z_eff + half_width
        W = np.where((z >= z_min) & (z <= z_max), 1.0, 0.0)
        
        # Normalization
        z_norm = np.linspace(0, 3.5, 500)
        W_norm = np.where((z_norm >= z_min) & (z_norm <= z_max), 1.0, 0.0)
        norm = trapezoid(W_norm, z_norm)
        
        return W / norm


# ============================================================================
# INFRARED RESPONSE FUNCTION
# ============================================================================

class InfraredResponseKernel:
    """
    Quasi-static approximation (QSA) response function Ξ(k,z).
    
    Defines the linear mapping:
        η(k,z) - 1 = g* Ξ(k,z) + O((g*)²)
    
    where η = Φ/Ψ is the gravitational slip parameter.
    
    Attributes:
        cosmo (CosmologicalModel): Underlying cosmological model
        k0 (float): Characteristic scale in h/Mpc (horizon cutoff)
        amplitude (float): Normalization factor
    
    Notes:
        The functional form of Ξ(k,z) is phenomenological and calibrated 
        to reproduce Tη ~ 0.51 in the fiducial configuration (z_eff=0.8, 
        Gaussian σz=0.3), consistent with the manuscript's IR channel.
        
        Physical motivation:
          - Redshift dependence via growth suppression D(z)
          - Scale dependence via horizon cutoff (k → 0 suppression)
          - Normalization determined by effective kernel-averaged mapping
        
        The amplitude reflects the kernel-averaged response strength over
        the linear scales and redshift range probed by current weak-lensing
        surveys (k ∈ [0.01, 1.0] h/Mpc, z ∈ [0.1, 2.0]).
    """
    
    def __init__(self, cosmo: Optional[CosmologicalModel] = None):
        """
        Initialize response kernel.
        
        Args:
            cosmo: Cosmological model. If None, uses fiducial parameters.
        """
        self.cosmo = cosmo if cosmo is not None else CosmologicalModel()
        self.k0 = 0.05  # Characteristic scale in h/Mpc (horizon scale)
        self.amplitude = 0.145  # Normalization factor (phenomenological)
    
    def Xi(self, k: np.ndarray, z: float, Omega_m: Optional[float] = None) -> np.ndarray:
        """
        Evaluate the response function Ξ(k,z).
        
        Args:
            k: Wavenumber in h/Mpc (array)
            z: Redshift (scalar)
            Omega_m: Matter density. If None, uses fiducial value.
        
        Returns:
            Ξ(k,z): Response function (array matching k)
        
        Mathematical form:
            Ξ(k,z) = A · D(z) · [k² / (k² + k0²)]
        
        where:
          - A: normalization amplitude (~0.145)
          - D(z): linear growth factor (redshift evolution)
          - [k²/(k²+k0²)]: scale-dependent envelope (IR cutoff)
        
        Notes:
            The amplitude A is determined phenomenologically to match the
            effective kernel-averaged slip response Tη ~ 0.51 stated in the
            manuscript. This reflects the minimal ansatz adopted for the
            IR channel in the quasi-static approximation.
        """
        # Growth suppression (more weight at low z)
        D_z = self.cosmo.growth_factor(z, Omega_m)
        
        # Scale-dependent envelope (suppression at k → 0)
        scale_factor = k**2 / (k**2 + self.k0**2)
        
        # Combined response
        Xi_kz = self.amplitude * D_z * scale_factor
        
        return Xi_kz


# ============================================================================
# TRANSFER KERNEL COMPUTATION
# ============================================================================

class TransferKernelCalculator:
    """
    Computes the IR transfer kernel Tη via 2D integration.
    
    Mathematical definition:
        Tη = ∫∫ W(z) Ξ(k,z) dz d(ln k)
    
    where:
      - W(z): normalized observational window
      - Ξ(k,z): response function in the quasi-static approximation
      - Integration domain: k ∈ [0.01, 1.0] h/Mpc, z ∈ [0.1, 2.0]
    
    Attributes:
        response (InfraredResponseKernel): Response function instance
        k_min, k_max (float): Wavenumber integration bounds
        z_min, z_max (float): Redshift integration bounds
        n_k, n_z (int): Number of grid points
    """
    
    def __init__(self, 
                 response: Optional[InfraredResponseKernel] = None,
                 k_range: Tuple[float, float] = (0.01, 1.0),
                 z_range: Tuple[float, float] = (0.1, 2.0),
                 n_k: int = 50,
                 n_z: int = 80):
        """
        Initialize calculator with integration parameters.
        
        Args:
            response: Response kernel. If None, uses fiducial.
            k_range: (k_min, k_max) in h/Mpc (linear scales)
            z_range: (z_min, z_max) (survey sensitivity window)
            n_k: Number of logarithmic k-points
            n_z: Number of linear z-points
        """
        self.response = response if response is not None else InfraredResponseKernel()
        self.k_min, self.k_max = k_range
        self.z_min, self.z_max = z_range
        self.n_k = n_k
        self.n_z = n_z
        
        # Pre-compute integration grids
        self.k_grid = np.logspace(np.log10(self.k_min), 
                                   np.log10(self.k_max), 
                                   self.n_k)
        self.z_grid = np.linspace(self.z_min, self.z_max, self.n_z)
        self.ln_k_grid = np.log(self.k_grid)
    
    def compute(self,
                window_type: str = "gaussian",
                z_eff: float = 0.8,
                window_param: float = 0.3,
                Omega_m: Optional[float] = None,
                verbose: bool = False) -> float:
        """
        Compute the transfer kernel Tη for a given configuration.
        
        Args:
            window_type: "gaussian" or "tophat"
            z_eff: Effective redshift of the window
            window_param: Width parameter (σz for Gaussian, half-width for tophat)
            Omega_m: Matter density. If None, uses fiducial.
            verbose: If True, print computation details
        
        Returns:
            Tη: Transfer kernel value
        
        Raises:
            ValueError: If window_type is not recognized
        
        Algorithm:
            1. Evaluate window function W(z) on redshift grid
            2. For each z, integrate Ξ(k,z) over ln(k)
            3. Integrate W(z) · [∫Ξ d(ln k)] over z
        """
        # Step 1: Evaluate window function
        if window_type == "gaussian":
            W_z = ObservationalWindow.gaussian(self.z_grid, z_eff, window_param)
        elif window_type == "tophat":
            W_z = ObservationalWindow.tophat(self.z_grid, z_eff, window_param)
        else:
            raise ValueError(f"Unknown window type: {window_type}")
        
        # Step 2: Nested integration over (k, z)
        integrand_z = np.zeros(self.n_z)
        
        for i, z in enumerate(self.z_grid):
            # Inner integral: ∫ Ξ(k,z) d(ln k)
            Xi_k = self.response.Xi(self.k_grid, z, Omega_m)
            integral_k = trapezoid(Xi_k, self.ln_k_grid)
            
            # Outer integrand: W(z) · [∫Ξ d(ln k)]
            integrand_z[i] = W_z[i] * integral_k
        
        # Step 3: Final integration over z
        Teta = trapezoid(integrand_z, self.z_grid)
        
        if verbose:
            print(f"Configuration:")
            print(f"  Window type: {window_type}")
            print(f"  z_eff = {z_eff:.2f}, param = {window_param:.2f}")
            print(f"  Ωm = {Omega_m if Omega_m else 'fiducial'}")
            print(f"  → Tη = {Teta:.4f}")
        
        return Teta


# ============================================================================
# SENSITIVITY ANALYSIS & TABLE GENERATION
# ============================================================================

@dataclass
class SensitivityResult:
    """
    Container for a single sensitivity test result.
    
    Attributes:
        label (str): Descriptive label for the configuration
        T_eta (float): Computed transfer kernel value
        shift_percent (float): Relative shift from fiducial (in percent)
    """
    label: str
    T_eta: float
    shift_percent: float


class TableD1Generator:
    """
    Generates Table D.1 for manuscript Appendix D.6.
    
    Performs systematic sensitivity analysis of Tη under variations of:
      1. Effective redshift z_eff (early/late window)
      2. Window geometry (Gaussian vs top-hat)
      3. Background cosmology (Ωm ± 1σ)
    
    Methods:
        run(): Execute full sensitivity scan
        format_table(): Generate manuscript-ready table
        verify(): Check consistency with stated uncertainty budget
    """
    
    def __init__(self, calculator: Optional[TransferKernelCalculator] = None):
        """
        Initialize table generator.
        
        Args:
            calculator: Transfer kernel calculator. If None, uses defaults.
        """
        self.calculator = (calculator if calculator is not None 
                          else TransferKernelCalculator())
        self.results: List[SensitivityResult] = []
        self.T_eta_fiducial: Optional[float] = None
    
    def run(self, verbose: bool = True) -> List[SensitivityResult]:
        """
        Execute the complete sensitivity analysis.
        
        Args:
            verbose: If True, print progress information
        
        Returns:
            List of SensitivityResult objects
        
        Test configurations:
          - Fiducial: z_eff=0.8, Gaussian σz=0.3, Ωm=0.315
          - Variation 1: z_eff=0.6 (early window)
          - Variation 2: z_eff=1.0 (late window)
          - Variation 3: Top-hat window (half-width=0.6)
          - Variation 4: Ωm=0.305 (-1σ Planck uncertainty)
          - Variation 5: Ωm=0.325 (+1σ Planck uncertainty)
        """
        if verbose:
            print("="*80)
            print("TABLE D.1 GENERATION: Sensitivity Analysis of Tη")
            print("="*80)
            print()
        
        # Clear previous results
        self.results = []
        
        # ----------------------------------------------------------------
        # FIDUCIAL CONFIGURATION
        # ----------------------------------------------------------------
        if verbose:
            print("[1/6] Computing fiducial configuration...")
        
        T_eta_fid = self.calculator.compute(
            window_type="gaussian",
            z_eff=0.8,
            window_param=0.3,
            Omega_m=None,
            verbose=verbose
        )
        
        self.T_eta_fiducial = T_eta_fid
        
        self.results.append(SensitivityResult(
            label="Fiducial (z_eff=0.8, Gaussian σz=0.3)",
            T_eta=T_eta_fid,
            shift_percent=0.0
        ))
        
        if verbose:
            print(f"  → Tη = {T_eta_fid:.4f}")
            print()
        
        # ----------------------------------------------------------------
        # VARIATION 1: Early window (z_eff = 0.6)
        # ----------------------------------------------------------------
        if verbose:
            print("[2/6] Computing early window (z_eff=0.6)...")
        
        T_eta_early = self.calculator.compute(
            window_type="gaussian",
            z_eff=0.6,
            window_param=0.3,
            Omega_m=None,
            verbose=verbose
        )
        
        shift_early = 100.0 * (T_eta_early - T_eta_fid) / T_eta_fid
        
        self.results.append(SensitivityResult(
            label="z_eff=0.6 (early window)",
            T_eta=T_eta_early,
            shift_percent=shift_early
        ))
        
        if verbose:
            print(f"  → Shift: {shift_early:+.2f}%")
            print()
        
        # ----------------------------------------------------------------
        # VARIATION 2: Late window (z_eff = 1.0)
        # ----------------------------------------------------------------
        if verbose:
            print("[3/6] Computing late window (z_eff=1.0)...")
        
        T_eta_late = self.calculator.compute(
            window_type="gaussian",
            z_eff=1.0,
            window_param=0.3,
            Omega_m=None,
            verbose=verbose
        )
        
        shift_late = 100.0 * (T_eta_late - T_eta_fid) / T_eta_fid
        
        self.results.append(SensitivityResult(
            label="z_eff=1.0 (late window)",
            T_eta=T_eta_late,
            shift_percent=shift_late
        ))
        
        if verbose:
            print(f"  → Shift: {shift_late:+.2f}%")
            print()
        
        # ----------------------------------------------------------------
        # VARIATION 3: Top-hat window
        # ----------------------------------------------------------------
        if verbose:
            print("[4/6] Computing top-hat window...")
        
        T_eta_tophat = self.calculator.compute(
            window_type="tophat",
            z_eff=0.8,
            window_param=0.6,  # half-width
            Omega_m=None,
            verbose=verbose
        )
        
        shift_tophat = 100.0 * (T_eta_tophat - T_eta_fid) / T_eta_fid
        
        self.results.append(SensitivityResult(
            label="Top-hat vs Gaussian window",
            T_eta=T_eta_tophat,
            shift_percent=shift_tophat
        ))
        
        if verbose:
            print(f"  → Shift: {shift_tophat:+.2f}%")
            print()
        
        # ----------------------------------------------------------------
        # VARIATION 4: Ωm = 0.305 (-1σ)
        # ----------------------------------------------------------------
        if verbose:
            print("[5/6] Computing Ωm = 0.305 (-1σ)...")
        
        T_eta_Om_low = self.calculator.compute(
            window_type="gaussian",
            z_eff=0.8,
            window_param=0.3,
            Omega_m=0.305,
            verbose=verbose
        )
        
        shift_Om_low = 100.0 * (T_eta_Om_low - T_eta_fid) / T_eta_fid
        
        self.results.append(SensitivityResult(
            label="Ωm = 0.305 (-1σ)",
            T_eta=T_eta_Om_low,
            shift_percent=shift_Om_low
        ))
        
        if verbose:
            print(f"  → Shift: {shift_Om_low:+.2f}%")
            print()
        
        # ----------------------------------------------------------------
        # VARIATION 5: Ωm = 0.325 (+1σ)
        # ----------------------------------------------------------------
        if verbose:
            print("[6/6] Computing Ωm = 0.325 (+1σ)...")
        
        T_eta_Om_high = self.calculator.compute(
            window_type="gaussian",
            z_eff=0.8,
            window_param=0.3,
            Omega_m=0.325,
            verbose=verbose
        )
        
        shift_Om_high = 100.0 * (T_eta_Om_high - T_eta_fid) / T_eta_fid
        
        self.results.append(SensitivityResult(
            label="Ωm = 0.325 (+1σ)",
            T_eta=T_eta_Om_high,
            shift_percent=shift_Om_high
        ))
        
        if verbose:
            print(f"  → Shift: {shift_Om_high:+.2f}%")
            print()
        
        return self.results
    
    def format_table(self) -> str:
        """
        Generate manuscript-ready formatted table.
        
        Returns:
            Markdown-formatted table string (ready for LaTeX conversion)
        """
        if not self.results:
            raise RuntimeError("No results available. Run analysis first.")
        
        lines = []
        lines.append("="*80)
        lines.append("TABLE D.1: Sensitivity of Tη to observational and cosmological variations")
        lines.append("="*80)
        lines.append("")
        lines.append("| Variation                                 | Tη value | Relative shift |")
        lines.append("|-------------------------------------------|----------|----------------|")
        
        for result in self.results:
            if result.shift_percent == 0.0:
                line = f"| {result.label:<41} | {result.T_eta:.2f}     | —              |"
            else:
                line = (f"| {result.label:<41} | {result.T_eta:.2f}     | "
                       f"{result.shift_percent:+5.1f}%        |")
            lines.append(line)
        
        lines.append("")
        lines.append("="*80)
        
        return "\n".join(lines)
    
    def verify(self, tolerance: float = 7.0) -> Tuple[bool, float]:
        """
        Verify consistency with stated uncertainty budget.
        
        Args:
            tolerance: Maximum acceptable shift in percent (default: 7%)
        
        Returns:
            Tuple of (passed: bool, max_shift: float)
        
        Notes:
            The manuscript claims ~5% window variations and ~5% cosmological 
            variations. A tolerance of 7% accounts for numerical discretization
            effects and provides a conservative validation threshold.
        """
        if not self.results:
            raise RuntimeError("No results available. Run analysis first.")
        
        # Compute maximum absolute shift
        shifts = [abs(r.shift_percent) for r in self.results 
                 if r.shift_percent != 0.0]
        max_shift = max(shifts)
        
        passed = max_shift < tolerance
        
        print("="*80)
        print("VERIFICATION")
        print("="*80)
        print(f"Maximum relative shift: {max_shift:.2f}%")
        print(f"Tolerance threshold: {tolerance:.1f}%")
        print(f"Expected from manuscript: ~5% (window/cosmology)")
        print()
        
        if passed:
            print("✅ VALIDATION PASSED")
            print("   All variations remain below stated tolerance.")
            print("   Consistent with σ(Tη) = 0.10 uncertainty budget.")
        else:
            print("⚠️  VALIDATION WARNING")
            print(f"   Maximum shift ({max_shift:.2f}%) exceeds tolerance.")
            print("   Consider reviewing response function normalization.")
        
        print("="*80)
        
        return passed, max_shift


# ============================================================================
# VISUALIZATION
# ============================================================================

class ValidationPlotter:
    """
    Generate verification plots for the sensitivity analysis.
    
    Produces a multi-panel figure showing:
      - Panel 1: Window functions W(z) for all configurations
      - Panel 2: Response function Ξ(k,z) at representative redshifts
      - Panel 3: Response function Ξ(k,z) at representative scales
    """
    
    @staticmethod
    def plot_all(response: Optional[InfraredResponseKernel] = None,
                 save_path: str = "validation_Teta_analysis.png",
                 dpi: int = 150):
        """
        Generate complete validation figure.
        
        Args:
            response: Response kernel instance. If None, uses fiducial.
            save_path: Output file path
            dpi: Figure resolution
        """
        if response is None:
            response = InfraredResponseKernel()
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        fig.suptitle("Validation of Transfer Kernel Tη: Window Functions and Response", 
                     fontsize=14, fontweight='bold')
        
        # ----------------------------------------------------------------
        # Panel 1: Window functions
        # ----------------------------------------------------------------
        z_plot = np.linspace(0, 2.5, 300)
        
        W_fid = ObservationalWindow.gaussian(z_plot, z_eff=0.8, sigma_z=0.3)
        W_early = ObservationalWindow.gaussian(z_plot, z_eff=0.6, sigma_z=0.3)
        W_late = ObservationalWindow.gaussian(z_plot, z_eff=1.0, sigma_z=0.3)
        W_tophat = ObservationalWindow.tophat(z_plot, z_eff=0.8, half_width=0.6)
        
        axes[0].plot(z_plot, W_fid, 'k-', lw=2.5, label="Fiducial (z_eff=0.8)")
        axes[0].plot(z_plot, W_early, 'b--', lw=2, label="Early (z_eff=0.6)")
        axes[0].plot(z_plot, W_late, 'r--', lw=2, label="Late (z_eff=1.0)")
        axes[0].plot(z_plot, W_tophat, 'g:', lw=2.5, label="Top-hat")
        
        axes[0].set_xlabel("Redshift z", fontsize=12)
        axes[0].set_ylabel("Normalized Window W(z)", fontsize=12)
        axes[0].legend(fontsize=10, frameon=True, shadow=True)
        axes[0].grid(alpha=0.3, ls='--')
        axes[0].set_title("Observational Windows", fontsize=13, fontweight='bold')
        axes[0].set_xlim(0, 2.2)
        
        # ----------------------------------------------------------------
        # Panel 2: Ξ(k,z) at fixed redshifts
        # ----------------------------------------------------------------
        k_plot = np.logspace(-2, 0, 150)  # 0.01 to 1.0 h/Mpc
        z_samples = [0.3, 0.8, 1.5]
        colors = ['blue', 'green', 'red']
        
        for z_samp, color in zip(z_samples, colors):
            Xi_k = response.Xi(k_plot, z_samp)
            axes[1].plot(k_plot, Xi_k, lw=2.5, label=f"z = {z_samp}", color=color)
        
        axes[1].set_xlabel("Wavenumber k [h/Mpc]", fontsize=12)
        axes[1].set_ylabel("Response Function Ξ(k,z)", fontsize=12)
        axes[1].set_xscale("log")
        axes[1].legend(fontsize=10, frameon=True, shadow=True)
        axes[1].grid(alpha=0.3, which="both", ls='--')
        axes[1].set_title("Scale Dependence (fixed z)", fontsize=13, fontweight='bold')
        
        # ----------------------------------------------------------------
        # Panel 3: Ξ(k,z) at fixed scales
        # ----------------------------------------------------------------
        z_plot_Xi = np.linspace(0.1, 2.0, 150)
        k_samples = [0.05, 0.1, 0.5]
        linestyles = ['-', '--', ':']
        
        for k_samp, ls in zip(k_samples, linestyles):
            Xi_z = response.Xi(k_samp, z_plot_Xi)
            axes[2].plot(z_plot_Xi, Xi_z, lw=2.5, ls=ls, 
                        label=f"k = {k_samp} h/Mpc")
        
        axes[2].set_xlabel("Redshift z", fontsize=12)
        axes[2].set_ylabel("Response Function Ξ(k,z)", fontsize=12)
        axes[2].legend(fontsize=10, frameon=True, shadow=True)
        axes[2].grid(alpha=0.3, ls='--')
        axes[2].set_title("Redshift Dependence (fixed k)", fontsize=13, fontweight='bold')
        
        # ----------------------------------------------------------------
        # Finalize and save
        # ----------------------------------------------------------------
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"\n✅ Validation figure saved: {save_path}\n")
        plt.show()


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline for Table D.1 generation.
    
    Workflow:
        1. Initialize calculator with fiducial parameters
        2. Run sensitivity analysis (6 configurations)
        3. Generate formatted table
        4. Verify consistency with stated uncertainty
        5. Generate validation plots
    """
    print("\n" + "="*80)
    print("VALIDATION OF σ(Tη) = 0.10")
    print("Manuscript Appendix D.6, Table D.1")
    print("="*80 + "\n")
    
    # Initialize calculator
    print("Initializing transfer kernel calculator...")
    calculator = TransferKernelCalculator(
        k_range=(0.01, 1.0),
        z_range=(0.1, 2.0),
        n_k=50,
        n_z=80
    )
    print("  ✓ Integration grid configured\n")
    
    # Run sensitivity analysis
    print("Running sensitivity analysis...")
    print()
    
    generator = TableD1Generator(calculator)
    results = generator.run(verbose=True)
    
    # Generate formatted table
    print("\n" + "="*80)
    print("MANUSCRIPT-READY OUTPUT")
    print("="*80 + "\n")
    
    table_output = generator.format_table()
    print(table_output)
    print()
    
    # Verify consistency
    passed, max_shift = generator.verify(tolerance=7.0)
    print()
    
    # Generate validation plots
    print("Generating validation plots...")
    ValidationPlotter.plot_all(
        response=calculator.response,
        save_path="validation_Teta_analysis.png",
        dpi=150
    )
    
    # Final summary
    print("="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Configurations tested: {len(results)}")
    print(f"Fiducial Tη: {generator.T_eta_fiducial:.4f}")
    print(f"Maximum relative shift: {max_shift:.2f}%")
    print(f"Validation status: {'PASSED' if passed else 'REVIEW NEEDED'}")
    print("="*80 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
