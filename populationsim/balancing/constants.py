# Constants

# Convergence constants
DEFAULT_MAX_ITERATIONS = 10000
MAX_DELTA = 1.0e-9
MAX_DELTA32 = 1.0e-5
MAX_GAMMA = 1.0e-5


MIN_GAMMA = 1.0e-10  # Used to avoid division by zero in the algorithm
IMPORTANCE_ADJUST = 2
IMPORTANCE_ADJUST_COUNT = 100
MIN_IMPORTANCE = 1.0
MAX_RELAXATION_FACTOR = 1000000
MIN_CONTROL_VALUE = 0.1
MAX_INT = 1 << 31

# delta to check for non-convergence without progress
ALT_MAX_DELTA = 1.0e-14
