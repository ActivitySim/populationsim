

class Integerizer(object):

    def __init__(self,
                 incidence_table,
                 control_totals,
                 control_importance_weights,
                 initial_weights,
                 final_weights,
                 relaxation_factors,
                 household_based_controls,
                 total_households_control_index,
                 debug_control_set):

        self.incidence_table = incidence_table
        self.control_totals = control_totals
        self.control_importance_weights = control_importance_weights
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.relaxation_factors = relaxation_factors
        self.household_based_controls = household_based_controls
        self.total_households_control_index = total_households_control_index
        self.is_household_control = None
        self.debug_control_set = debug_control_set


def do_integerizing(label,
                    id,
                    incidence_table,
                    control_totals,
                    control_importance_weights,
                    initial_weights,
                    final_weights,
                    relaxation_factors,
                    household_based_controls,
                    total_households_control_index,
                    debug_control_set):

    sample_count = len(incidence_table.index)
    control_count = len(incidence_table.columns)

    # if the incidence table has only one record, then the final integer weights
    # should be just an array with 1 element equal to the total number of households;
    if sample_count <= 1:
        integer_weights = control_totals
        return integer_weights

    # otherwise, solve for the integer weights using the Mixed Integer Programming solver.

    integerizer = Integerizer(incidence_table=incidence_table,
                              control_totals=control_totals,
                              control_importance_weights=control_importance_weights,
                              initial_weights=initial_weights,
                              final_weights=final_weights,
                              relaxation_factors=relaxation_factors,
                              household_based_controls=household_based_controls,
                              total_households_control_index=total_households_control_index,
                              debug_control_set=debug_control_set)

    return True
