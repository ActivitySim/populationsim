import numpy as np


def quantize_weights(weights, quantum):
    """Snap weights to an absolute grid when quantization is enabled."""
    values = np.asarray(weights, dtype=np.float64)

    if quantum in (None, False, 0, 0.0):
        return values
    if isinstance(quantum, bool):
        raise ValueError("INTEGERIZER_QUANTUM must be a positive finite number")

    try:
        quantum = float(quantum)
    except (TypeError, ValueError) as err:
        raise ValueError(
            "INTEGERIZER_QUANTUM must be a positive finite number"
        ) from err
    if not np.isfinite(quantum) or quantum <= 0:
        raise ValueError("INTEGERIZER_QUANTUM must be a positive finite number")

    quantized = np.rint(values / quantum) * quantum
    # Zero has structural meaning in the integerizers: it makes a household
    # ineligible for selection. Quantization must not turn an eligible,
    # positive household into an ineligible one. Tiny positive values share
    # one canonical sentinel; the existing log overflow guard maps it to the
    # same finite objective coefficient in every case.
    positive_to_zero = (values > 0) & (quantized == 0)
    quantized[positive_to_zero] = np.nextafter(0.0, 1.0)
    if not np.isfinite(quantized).all():
        raise ValueError("INTEGERIZER_QUANTUM produced non-finite weights")

    return quantized
