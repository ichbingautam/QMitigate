"""
Zero-Noise Extrapolation (ZNE) Module

Implements error mitigation via zero-noise extrapolation as described in:
- Temme, K., Bravyi, S., & Gambetta, J. M. (2017). "Error mitigation for
  short-depth quantum circuits." Physical Review Letters.
- Giurgica-Tiron, T., et al. (2020). "Digital zero noise extrapolation for
  quantum error mitigation." IEEE International Conference on Quantum Computing.

The key insight: If we can scale the noise level λ, we can measure results
at λ = 1, 3, 5, ... and extrapolate back to λ = 0 (zero noise).
"""

from typing import List, Tuple, Callable, Optional
import numpy as np
from scipy.optimize import curve_fit


def richardson_extrapolate(
    scale_factors: List[float],
    expectation_values: List[float]
) -> float:
    """
    Richardson extrapolation to estimate zero-noise expectation value.

    Given noisy measurements at different noise scales, fit a polynomial
    and extrapolate to the y-intercept (noise = 0).

    Mathematical basis:
        E(λ) = E(0) + a₁λ + a₂λ² + ...

    We fit to the measured points and extract E(0).

    Args:
        scale_factors: List of noise scale factors [1, 3, 5, ...]
        expectation_values: Measured expectation values at each scale

    Returns:
        Extrapolated zero-noise expectation value

    Example:
        >>> scales = [1, 3, 5]
        >>> noisy_values = [0.8, 0.6, 0.45]
        >>> zero_noise = richardson_extrapolate(scales, noisy_values)
        >>> print(f"Mitigated: {zero_noise:.4f}")  # Should be ~0.95
    """
    if len(scale_factors) != len(expectation_values):
        raise ValueError("Scale factors and expectation values must have same length")

    if len(scale_factors) < 2:
        raise ValueError("Need at least 2 data points for extrapolation")

    # Fit polynomial of degree (n_points - 1)
    degree = len(scale_factors) - 1
    coeffs = np.polyfit(scale_factors, expectation_values, degree)

    # The value at x=0 is the last coefficient (constant term)
    return float(coeffs[-1])


def linear_extrapolate(
    scale_factors: List[float],
    expectation_values: List[float]
) -> Tuple[float, float]:
    """
    Simple linear extrapolation (first order Richardson).

    Assumes E(λ) = E(0) + a·λ

    Returns:
        Tuple of (zero_noise_estimate, slope)
    """
    if len(scale_factors) < 2:
        raise ValueError("Need at least 2 points")

    # Use only first two points for linear fit
    λ1, λ2 = scale_factors[0], scale_factors[1]
    E1, E2 = expectation_values[0], expectation_values[1]

    # Linear interpolation formula for E(0)
    slope = (E2 - E1) / (λ2 - λ1)
    E0 = E1 - slope * λ1

    return float(E0), float(slope)


def exponential_extrapolate(
    scale_factors: List[float],
    expectation_values: List[float],
    asymptote: float = 0.0
) -> Tuple[float, dict]:
    """
    Exponential decay extrapolation.

    Assumes E(λ) = a·exp(-b·λ) + c
    where c is the asymptotic value as noise → ∞

    This model is often more physically accurate as it captures
    the exponential decay of quantum coherence.

    Args:
        scale_factors: Noise scale factors
        expectation_values: Measured values
        asymptote: Fixed asymptotic value (default 0 for Z expectation)

    Returns:
        Tuple of (zero_noise_estimate, fit_params)
    """
    def exp_model(x, a, b):
        return a * np.exp(-b * np.array(x)) + asymptote

    try:
        popt, _ = curve_fit(
            exp_model,
            scale_factors,
            expectation_values,
            p0=[1.0, 0.1],
            maxfev=1000
        )
        a, b = popt
        zero_noise = a + asymptote  # exp(0) = 1
        return float(zero_noise), {"amplitude": a, "decay_rate": b}
    except RuntimeError:
        # Fall back to linear if exponential fit fails
        E0, _ = linear_extrapolate(scale_factors, expectation_values)
        return E0, {"fallback": "linear"}


class ZeroNoiseExtrapolator:
    """
    Zero-Noise Extrapolation engine for error mitigation.

    This class orchestrates the full ZNE workflow:
    1. Run the circuit at multiple noise scales (1×, 3×, 5×, ...)
    2. Fit the noisy results to a model
    3. Extrapolate to zero noise

    Reference Implementation of:
    - Digital folding: U → U U† U (scales noise by 3×)
    - Richardson extrapolation: polynomial fit to data points

    Example:
        >>> from qmitigate import Circuit, Simulator, create_depolarizing_noise_model
        >>>
        >>> # Create a noisy simulator
        >>> noise = create_depolarizing_noise_model(0.01)
        >>> sim = Simulator(noise)
        >>>
        >>> # Build a circuit
        >>> circuit = Circuit(2)
        >>> circuit.h(0)
        >>> circuit.cnot(0, 1)
        >>>
        >>> # Run ZNE
        >>> zne = ZeroNoiseExtrapolator(sim, scale_factors=[1, 3, 5])
        >>> ideal_value, mitigated_value = zne.mitigate_expectation_Z(circuit, qubit=0)
    """

    def __init__(
        self,
        simulator: "Simulator",
        scale_factors: List[int] = None,
        extrapolation_method: str = "richardson",
        shots_per_scale: int = 1000
    ):
        """
        Initialize the ZNE engine.

        Args:
            simulator: QMitigate Simulator instance with noise model
            scale_factors: Noise scale factors (must be odd integers)
            extrapolation_method: "richardson", "linear", or "exponential"
            shots_per_scale: Number of measurement shots per scale factor
        """
        self.simulator = simulator
        self.scale_factors = scale_factors or [1, 3, 5]
        self.extrapolation_method = extrapolation_method
        self.shots_per_scale = shots_per_scale

        # Validate scale factors
        for s in self.scale_factors:
            if s < 1 or s % 2 == 0:
                raise ValueError(f"Scale factors must be odd positive integers, got {s}")

    def mitigate_expectation_Z(
        self,
        circuit: "Circuit",
        qubit: int,
        return_raw: bool = False
    ) -> Tuple[float, Optional[dict]]:
        """
        Compute error-mitigated ⟨Z⟩ expectation value.

        Args:
            circuit: The quantum circuit to execute
            qubit: Qubit index for Z measurement
            return_raw: If True, return dict with all intermediate data

        Returns:
            Mitigated expectation value (and optionally raw data)
        """
        from .qmitigate_cpp import fold_circuit_global

        noisy_values = []

        for scale in self.scale_factors:
            # Digitally fold the circuit to scale the noise
            folded_circuit = fold_circuit_global(circuit, scale)

            # Run and measure
            total = 0.0
            for _ in range(self.shots_per_scale):
                state = self.simulator.run(folded_circuit)
                total += state.expectation_Z(qubit)

            avg = total / self.shots_per_scale
            noisy_values.append(avg)

        # Extrapolate to zero noise
        if self.extrapolation_method == "richardson":
            mitigated = richardson_extrapolate(
                [float(s) for s in self.scale_factors],
                noisy_values
            )
        elif self.extrapolation_method == "linear":
            mitigated, _ = linear_extrapolate(
                [float(s) for s in self.scale_factors],
                noisy_values
            )
        elif self.extrapolation_method == "exponential":
            mitigated, _ = exponential_extrapolate(
                [float(s) for s in self.scale_factors],
                noisy_values
            )
        else:
            raise ValueError(f"Unknown extrapolation method: {self.extrapolation_method}")

        if return_raw:
            return mitigated, {
                "scale_factors": self.scale_factors,
                "noisy_values": noisy_values,
                "unmitigated": noisy_values[0],
                "improvement": abs(mitigated - 1.0) < abs(noisy_values[0] - 1.0)
            }

        return mitigated, None

    def analyze_bias_variance(
        self,
        circuit: "Circuit",
        qubit: int,
        ideal_value: float,
        num_trials: int = 100
    ) -> dict:
        """
        Analyze the bias-variance tradeoff of ZNE.

        As noted in the Mitiq paper, extrapolation reduces bias but can
        increase variance. This method quantifies both.

        Args:
            circuit: Circuit to analyze
            qubit: Measurement qubit
            ideal_value: True noiseless expectation value
            num_trials: Number of independent ZNE runs

        Returns:
            Dictionary with bias, variance, and MSE for mitigated vs unmitigated
        """
        mitigated_values = []
        unmitigated_values = []

        for _ in range(num_trials):
            mitigated, raw = self.mitigate_expectation_Z(circuit, qubit, return_raw=True)
            mitigated_values.append(mitigated)
            unmitigated_values.append(raw["unmitigated"])

        mitigated_arr = np.array(mitigated_values)
        unmitigated_arr = np.array(unmitigated_values)

        return {
            "mitigated": {
                "mean": float(np.mean(mitigated_arr)),
                "std": float(np.std(mitigated_arr)),
                "bias": float(np.mean(mitigated_arr) - ideal_value),
                "variance": float(np.var(mitigated_arr)),
                "mse": float(np.mean((mitigated_arr - ideal_value) ** 2))
            },
            "unmitigated": {
                "mean": float(np.mean(unmitigated_arr)),
                "std": float(np.std(unmitigated_arr)),
                "bias": float(np.mean(unmitigated_arr) - ideal_value),
                "variance": float(np.var(unmitigated_arr)),
                "mse": float(np.mean((unmitigated_arr - ideal_value) ** 2))
            }
        }
