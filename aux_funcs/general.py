import numpy as np


def V_D_Lhermitte_1988(d_mm: float) -> float:
    d_mm *= 1e-3
    return 9.25 * (1 - np.exp(-1 * (68000 * (d_mm**2) + 488 * d_mm)))
