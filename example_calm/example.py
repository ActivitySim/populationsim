import pandas as pd

from populationsim.balancer import list_balancer

# rows are elements for which factors are calculated, columns are constraints to be satisfied
incidence_table = pd.DataFrame({
    'hh': [1, 1, 1, 1, 1, 1, 1, 1],
    'hh_1': [1, 1, 1, 0, 0, 0, 0, 0],
    'hh_2': [0, 0, 0, 1, 1, 1, 1, 1],
    'p1': [1, 1, 2, 1, 0, 1, 2, 1],
    'p2': [1, 0, 1, 0, 2, 1, 1, 1],
    'p3': [1, 1, 0, 2, 1, 0, 2, 0],
})

# column totals which the final weighted incidence table sums must satisfy
constraints = [100, 35, 65, 91, 65, 104]


# one weight per row in incidence table (or a scalar if all initial weights are the same)
initial_weights = 20

control_importance_weights = [10000000, 100000, 100000, 100000, 100000, 100000]

weights, controls, status = list_balancer(
    incidence_table=incidence_table,
    constraints=constraints,
    initial_weights=initial_weights,
    control_importance_weights=control_importance_weights,
    master_control_index=0,
    max_iterations=10000)


for key, value in status.iteritems():
    print "%s: %s" % (key, value)

print "\nweights\n", weights

print "\ncontrols\n", controls
