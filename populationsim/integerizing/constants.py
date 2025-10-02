STATUS_OPTIMAL = "OPTIMAL"
STATUS_FEASIBLE = "FEASIBLE"
STATUS_SUCCESS = [STATUS_OPTIMAL, STATUS_FEASIBLE]

# 'CBC', 'GLPK_MI', 'ECOS_BB'
CVX_SOLVER = "GLPK_MI"

# Order of vectorization for cvxpy
# 'C' for C-style row-major order, 'F' for Fortran-style column-major order
# Note: cvxpy is deprecating 'F' order, so we use 'C' order.
ORDER = "C"
