import time

import pandas as pd

# import pyximport; pyximport.install()
# from populationsim.cbalancer import clist_balancer

from populationsim.balancer import list_balancer
from populationsim.integerizer import do_integerizing


def print_elapsed_time(msg=None, t0=None, debug=False):
    t1 = time.time()
    if msg:
        t = t1 - (t0 or t1)
        msg = "Time to execute %s : %s seconds (%s minutes)" % (msg, round(t, 3), round(t/60.0))
        print msg
    return t1

# rows are elements for which factors are calculated, columns are constraints to be satisfied
incidence_table = pd.DataFrame({
    'hh': [1, 1, 1, 1, 1, 1, 1, 1],
    'hh_1': [1, 1, 1, 0, 0, 0, 0, 0],
    'hh_2': [0, 0, 0, 1, 1, 1, 1, 1],
    'p1': [1, 1, 2, 1, 0, 1, 2, 1],
    'p2': [1, 0, 1, 0, 2, 1, 1, 1],
    'p3': [1, 1, 0, 2, 1, 0, 2, 0],
})

household_based_controls = [True, True, True, False, False, False]
master_control_index = 0

# column totals which the final weighted incidence table sums must satisfy
control_totals = [100, 35, 65, 91, 65, 104]


# one weight per row in incidence table (or a scalar if all initial weights are the same)
initial_weights = 20

control_importance_weights = [10000000, 100000, 100000, 100000, 100000, 100000]


t0 = print_elapsed_time()

weights, controls, status = list_balancer(
    incidence_table=incidence_table,
    control_totals=control_totals,
    initial_weights=initial_weights,
    control_importance_weights=control_importance_weights,
    master_control_index=master_control_index,
    max_iterations=10000)

t0 = print_elapsed_time("list_balancer", t0)

for key, value in status.iteritems():
    print "%s: %s" % (key, value)

print "\nweights\n", weights

print "\ncontrols\n", controls


t0 = print_elapsed_time()

final_weights = weights.final

integer_weights = \
    do_integerizing(
        label='label',
        id=0,
        incidence_table=incidence_table,
        control_totals=control_totals,
        control_importance_weights=control_importance_weights,
        initial_weights=initial_weights,
        final_weights=weights.final,
        relaxation_factors=controls.relaxation_factor,
        household_based_controls=household_based_controls,
        total_households_control_index=master_control_index,
        debug_control_set=True)

print "\ninteger_weights\n", integer_weights

t0 = print_elapsed_time("integerizer", t0)
