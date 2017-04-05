import pandas as pd

from balancer import list_balancer
#
# # rows are elements for which factors are calculated, columns are constraints to be satisfied
# incidence_table = pd.DataFrame({
#     'hh': [1, 1, 1, 1, 1],
#     'hh_1': [1, 0, 0, 0, 0],
#     'hh_2': [0, 1, 0, 0, 0],
#     'hh_3': [0, 0, 1, 0, 0],
#     'hh_4': [0, 0, 0, 1, 1],
#     'p_0_15': [0, 1, 0, 0, 1],
#     'p_16_35': [0, 1, 1, 2, 3],
#     'p_36_64': [0, 0, 2, 2, 2],
#
# # column totals which the final weighted incidence table sums must satisfy
# constraints = [850,100,200,250,300,400,400,650,250]
#
# control_importance_weights = [1000, 1, 1, 1, 1, 1, 1, 1, 1]



# rows are elements for which factors are calculated, columns are constraints to be satisfied
incidence_table = pd.DataFrame({
    'hh_1': [1,1,1,0,0,0,0,0],
    'hh_2': [0,0,0,1,1,1,1,1],
    'p1': [1,1,2,1,0,1,2,1],
    'p2': [1,0,1,0,2,1,1,1],
    'p3': [1,1,0,2,1,0,2,0],
})

# one weight per row in incidence table
initial_weights = [1,1,1,1,1,1,1,1]

# column totals which the final weighted incidence table sums must satisfy
constraints = [35,65,91,65,104]

control_importance_weights = 1

list_balancer(incidence_table=incidence_table,
              constraints=constraints,
              initial_weights=initial_weights,
              control_importance_weights=control_importance_weights,
              master_control_index=None)






