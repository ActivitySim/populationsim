"""
Reference implementation and synthetic problems for the OR-Tools integerizers.

legacy_integerizer_ortools and legacy_simul_integerizer_ortools are the
per-element pywraplp model construction from populationsim/integerizing/lp_ortools.py
before it built MPModelProto messages with numpy. They are kept verbatim so the
tests and scripts/benchmark_integerizer.py can prove the new construction
produces byte-identical models and identical solutions.
"""

import contextlib
import time

import numpy as np

from populationsim.integerizing.constants import (
    STATUS_OPTIMAL,
    STATUS_FEASIBLE,
    STATUS_SUCCESS,
)


def legacy_integerizer_ortools(
    incidence,
    resid_weights,
    log_resid_weights,
    control_importance_weights,
    total_hh_control_index,
    lp_right_hand_side,
    relax_ge_upper_bound,
    hh_constraint_ge_bound,
    timeout_in_seconds,
):
    """Single integerizer as built before the MPModelProto rewrite."""

    from ortools.linear_solver import pywraplp

    STATUS_TEXT = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }

    control_count, sample_count = incidence.shape

    # - Instantiate a mixed-integer solver
    solver = pywraplp.Solver(
        "IntegerizeCbc", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
    )

    # Safe variable construction
    x = [None] * sample_count
    relax_le = [None] * control_count
    relax_ge = [None] * control_count
    hh_constraint_le = [None] * control_count
    hh_constraint_ge = [None] * control_count

    # - Create binary integer variables
    for hh in range(0, sample_count):
        # max_x == 0.0 if float_weights is an int, otherwise 1.0
        max_x = 1.0 - (resid_weights[hh] == 0.0)
        x[hh] = solver.NumVar(0.0, max_x, "x_" + str(hh))

    # - Create positive continuous constraint relaxation variables
    for c in range(0, control_count):
        # no relaxation for total households control
        if c != total_hh_control_index:
            relax_le[c] = solver.NumVar(
                0.0, lp_right_hand_side[c], "relax_le_" + str(c)
            )
            relax_ge[c] = solver.NumVar(
                0.0, relax_ge_upper_bound[c], "relax_ge_" + str(c)
            )

    # - Set objective function coefficients
    # use negative for objective and positive for relaxation penalties since solver is minimizing
    # objective = solver.Objective()
    # for hh in range(sample_count):
    #     objective.SetCoefficient(x[hh], -1.0 * log_resid_weights[hh])
    # for c in range(control_count):
    #     if c != total_hh_control_index:
    #         objective.SetCoefficient(relax_le[c], control_importance_weights[c])
    #         objective.SetCoefficient(relax_ge[c], control_importance_weights[c])

    epsilon = 1e-6  # Tiny weight tweak to break symmetry in non-unique cases
    z = (
        solver.Sum(
            x[hh] * (log_resid_weights[hh] + epsilon * hh)  # forces unique preference
            for hh in range(sample_count)
        )
        - solver.Sum(
            relax_le[c] * control_importance_weights[c]
            for c in range(control_count)
            if c != total_hh_control_index
        )
        - solver.Sum(
            relax_ge[c] * control_importance_weights[c]
            for c in range(control_count)
            if c != total_hh_control_index
        )
    )

    # - Set objective function
    objective = solver.Maximize(z)  # noqa: F841

    # - inequality constraints
    for c in range(0, control_count):
        # don't add inequality constraints for total households control
        if c == total_hh_control_index:
            continue
        # add the lower bound relaxation inequality constraint
        hh_constraint_le[c] = solver.Constraint(0, lp_right_hand_side[c])
        for hh in range(0, sample_count):
            hh_constraint_le[c].SetCoefficient(x[hh], incidence[c, hh])
            hh_constraint_le[c].SetCoefficient(relax_le[c], -1.0)

        # add the upper bound relaxation inequality constraint
        hh_constraint_ge[c] = solver.Constraint(
            lp_right_hand_side[c], hh_constraint_ge_bound[c]
        )
        for hh in range(0, sample_count):
            hh_constraint_ge[c].SetCoefficient(x[hh], incidence[c, hh])
            hh_constraint_ge[c].SetCoefficient(relax_ge[c], 1.0)

    # using Add and Sum is easier to read but a lot slower
    # for c in range(control_count):
    #     if c == total_hh_control_index:
    #         continue
    #     solver.Add(solver.Sum(x[hh]*incidence[c, hh] for hh in range(sample_count)) - relax_le[c]
    #                >= 0)
    #     solver.Add(solver.Sum(x[hh]*incidence[c, hh] for hh in range(sample_count)) - relax_le[c]
    #                <= lp_right_hand_side[c])
    #     solver.Add(solver.Sum(x[hh]*incidence[c, hh] for hh in range(sample_count)) + relax_ge[c]
    #                >= lp_right_hand_side[c])
    #     solver.Add(solver.Sum(x[hh]*incidence[c, hh] for hh in range(sample_count)) + relax_ge[c]
    #                <= hh_constraint_ge_bound[c])

    # - equality constraint for the total households control
    total_hh_constraint = lp_right_hand_side[total_hh_control_index]
    constraint_eq = solver.Constraint(total_hh_constraint, total_hh_constraint)
    for hh in range(0, sample_count):
        constraint_eq.SetCoefficient(x[hh], 1.0)

    solver.set_time_limit(timeout_in_seconds * 1000)

    solver.EnableOutput()

    result_status = solver.Solve()

    status_text = STATUS_TEXT[result_status]

    if status_text in STATUS_SUCCESS:
        resid_weights_out = np.asanyarray([x.solution_value() for x in x]).astype(
            np.float64
        )
    else:
        resid_weights_out = resid_weights

    return resid_weights_out, status_text


def legacy_simul_integerizer_ortools(
    sub_int_weights,
    parent_countrol_importance,
    parent_relax_ge_upper_bound,
    sub_control_importance,
    sub_float_weights,
    sub_resid_weights,
    lp_right_hand_side,
    parent_hh_constraint_ge_bound,
    sub_incidence,
    parent_incidence,
    total_hh_right_hand_side,
    relax_ge_upper_bound,
    parent_lp_right_hand_side,
    hh_constraint_ge_bound,
    parent_resid_weights,
    total_hh_sub_control_index,
    total_hh_parent_control_index,
    timeout_in_seconds,
):
    """Simul integerizer as built before the MPModelProto rewrite."""

    from ortools.linear_solver import pywraplp

    STATUS_TEXT = {
        pywraplp.Solver.OPTIMAL: STATUS_OPTIMAL,
        pywraplp.Solver.FEASIBLE: STATUS_FEASIBLE,
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }

    sample_count, sub_control_count = sub_incidence.shape
    _, parent_control_count = parent_incidence.shape
    sub_zone_count, _ = sub_float_weights.shape

    # setting indexes to -1 prevents creation of hh_controls relaxation variables
    # setting hh_control importance to zero eliminates them from the objective function
    # the latter approach is used by the cvx version
    # total_hh_sub_control_index = -1
    sub_control_importance[total_hh_sub_control_index] = 0

    # FIXME total_hh_parent_control_index should not exist???
    if total_hh_parent_control_index > 0:
        parent_countrol_importance[total_hh_parent_control_index] = 0

    # - Instantiate a mixed-integer solver
    solver = pywraplp.Solver(
        "SimulIntegerizeCbc", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
    )
    solver.EnableOutput()
    solver.set_time_limit(timeout_in_seconds * 1000)

    # x_max is 1.0 unless resid_weights is zero, in which case constrain x to 0.0
    x_max = (~(sub_float_weights == sub_int_weights)).astype(float)

    # - Create resid weight variables
    x = {}
    for z in range(sub_zone_count):
        for hh in range(sample_count):
            x[z, hh] = solver.NumVar(0.0, x_max[z, hh], "x[%s,%s]" % (z, hh))

    # - Create positive continuous constraint relaxation variables
    relax_le = {}
    relax_ge = {}
    for z in range(sub_zone_count):
        for c in range(sub_control_count):
            # no relaxation for total households control
            if c == total_hh_sub_control_index:
                continue
            relax_le[z, c] = solver.NumVar(
                0.0, lp_right_hand_side[z, c], "relax_le[%s,%s]" % (z, c)
            )
            relax_ge[z, c] = solver.NumVar(
                0.0, relax_ge_upper_bound[z, c], "relax_ge[%s,%s]" % (z, c)
            )

    parent_relax_le = {}
    parent_relax_ge = {}
    for c in range(parent_control_count):
        parent_relax_le[c] = solver.NumVar(
            0.0, parent_lp_right_hand_side[c], "parent_relax_le[%s]" % c
        )
        parent_relax_ge[c] = solver.NumVar(
            0.0, parent_relax_ge_upper_bound[c], "parent_relax_ge[%s]" % c
        )

    LOG_OVERFLOW = -725
    log_resid_weights = np.log(np.maximum(sub_resid_weights, np.exp(LOG_OVERFLOW)))
    assert not np.isnan(log_resid_weights).any()

    log_parent_resid_weights = np.log(
        np.maximum(parent_resid_weights, np.exp(LOG_OVERFLOW))
    )
    assert not np.isnan(log_parent_resid_weights).any()

    # - Set objective function coefficients
    z = (
        solver.Sum(
            x[z, hh] * log_resid_weights[z, hh]
            for z in range(sub_zone_count)
            for hh in range(sample_count)
        )
        + solver.Sum(
            x[z, hh] * log_parent_resid_weights[hh]
            for hh in range(sample_count)
            for z in range(sub_zone_count)
        )
        - solver.Sum(
            relax_le[z, c] * sub_control_importance[c]
            for z in range(sub_zone_count)
            for c in range(sub_control_count)
            if c != total_hh_sub_control_index
        )
        - solver.Sum(
            relax_ge[z, c] * sub_control_importance[c]
            for z in range(sub_zone_count)
            for c in range(sub_control_count)
            if c != total_hh_sub_control_index
        )
        - solver.Sum(
            parent_relax_le[c] * parent_countrol_importance[c]
            for c in range(parent_control_count)
        )
        - solver.Sum(
            parent_relax_ge[c] * parent_countrol_importance[c]
            for c in range(parent_control_count)
        )
    )

    # - Set objective function
    objective = solver.Maximize(z)  # noqa: F841

    # - sub inequality constraints
    sub_constraint_ge = {}
    sub_constraint_le = {}
    for z in range(sub_zone_count):
        for c in range(sub_control_count):

            # don't add inequality constraints for total households control
            if c == total_hh_sub_control_index:
                continue

            sub_constraint_le[z, c] = solver.Constraint(0, lp_right_hand_side[z, c])
            for hh in range(sample_count):
                sub_constraint_le[z, c].SetCoefficient(x[z, hh], sub_incidence[hh, c])
                sub_constraint_le[z, c].SetCoefficient(relax_le[z, c], -1.0)

            sub_constraint_ge[z, c] = solver.Constraint(
                lp_right_hand_side[z, c], hh_constraint_ge_bound[z, c]
            )
            for hh in range(sample_count):
                sub_constraint_ge[z, c].SetCoefficient(x[z, hh], sub_incidence[hh, c])
                sub_constraint_ge[z, c].SetCoefficient(relax_ge[z, c], 1.0)

    # - equality constraint for the total households control
    constraint_eq = {}
    for z in range(sub_zone_count):
        total_hh_constraint = total_hh_right_hand_side[z]

        constraint_eq[z] = solver.Constraint(total_hh_constraint, total_hh_constraint)
        for hh in range(sample_count):
            constraint_eq[z].SetCoefficient(x[z, hh], 1.0)

    # - sub inequality constraints
    parent_constraint_le = {}
    parent_constraint_ge = {}
    for c in range(parent_control_count):

        if c == total_hh_parent_control_index:
            continue

        parent_constraint_le[c] = solver.Constraint(0, parent_lp_right_hand_side[c])
        parent_constraint_ge[c] = solver.Constraint(
            parent_lp_right_hand_side[c], parent_hh_constraint_ge_bound[c]
        )

        for z in range(sub_zone_count):
            for hh in range(sample_count):
                parent_constraint_le[c].SetCoefficient(
                    x[z, hh], parent_incidence[hh, c]
                )
                parent_constraint_le[c].SetCoefficient(parent_relax_le[c], -1.0)

                parent_constraint_ge[c].SetCoefficient(
                    x[z, hh], parent_incidence[hh, c]
                )
                parent_constraint_ge[c].SetCoefficient(parent_relax_ge[c], 1.0)

    result_status = solver.Solve()

    status_text = STATUS_TEXT[result_status]

    if status_text in STATUS_SUCCESS:
        resid_weights_out = np.zeros(sub_resid_weights.shape)

        for z in range(sub_zone_count):
            for hh in range(sample_count):
                resid_weights_out[z, hh] = x[z, hh].solution_value()

        resid_weights_out = resid_weights_out.astype(np.float64)
    else:
        resid_weights_out = sub_resid_weights

    return resid_weights_out, status_text


def single_problem(
    sample_count,
    control_count=12,
    total_hh_control_index=0,
    zero_resid_share=0.05,
    backstopped=True,
    objective_dtype=np.float64,
    seed=0,
):
    """
    Synthetic np_integerizer_ortools kwargs, derived as Integerizer.integerize does.

    Controls mix one-hot household categories, person counts and an all-zero
    control. backstopped=False drops the person-count controls (as the
    unbackstopped retry does) and pins the total households value off the
    weighted sum. objective_dtype applies to the inputs that only feed the
    objective (the legacy code accepts nothing but doubles elsewhere).
    """
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < control_count - 2:
        k = rng.integers(0, 4, sample_count)
        rows += [(k == j).astype(float) for j in range(4)]
        if backstopped:
            rows.append(rng.poisson(0.7, sample_count).astype(float))
    rows = rows[: control_count - 2] + [np.zeros(sample_count)]
    rows.insert(total_hh_control_index, np.ones(sample_count))
    incidence = np.array(rows)

    float_weights = rng.gamma(2.0, 4.0, sample_count) + 0.01
    float_weights = np.rint(float_weights / 1e-6) * 1e-6
    whole = rng.random(sample_count) < zero_resid_share
    float_weights[whole] = np.ceil(float_weights[whole])

    relaxed_control_totals = np.round(float_weights @ incidence.T)
    total_hh_control_value = relaxed_control_totals[total_hh_control_index]
    if not backstopped:
        total_hh_control_value += 3

    int_weights = float_weights.astype(int)
    resid_weights = float_weights % 1.0
    lp_right_hand_side = np.maximum(
        relaxed_control_totals - int_weights @ incidence.T, 0
    )
    max_incidence_value = np.amax(incidence, axis=1)
    num_households = relaxed_control_totals[total_hh_control_index]
    importance = rng.choice([0.0, 10.0, 1000.0, 5000.0], len(incidence))

    return dict(
        incidence=incidence,
        resid_weights=resid_weights.astype(objective_dtype),
        log_resid_weights=np.log(np.maximum(resid_weights, np.exp(-725))).astype(
            objective_dtype
        ),
        control_importance_weights=importance.astype(objective_dtype),
        total_hh_control_index=total_hh_control_index,
        lp_right_hand_side=lp_right_hand_side,
        relax_ge_upper_bound=np.maximum(
            max_incidence_value * num_households - lp_right_hand_side, 0
        ),
        hh_constraint_ge_bound=np.maximum(
            total_hh_control_value * max_incidence_value, lp_right_hand_side
        ),
        timeout_in_seconds=60,
    )


def simul_problem(
    sample_count,
    sub_zone_count,
    sub_control_count=9,
    parent_control_count=8,
    total_hh_parent_control_index=-1,
    zero_resid_share=0.3,
    objective_dtype=np.float64,
    seed=0,
):
    """
    Synthetic np_simul_integerizer_ortools args, derived as
    SimulIntegerizer.integerize does. Includes an all-zero sub control.
    total_hh_parent_control_index > 0 exercises the parent total households
    control (its importance is zeroed and it gets no constraints).
    objective_dtype applies to the inputs that only feed the objective.
    """
    rng = np.random.default_rng(seed)
    sub = [np.ones(sample_count)]
    while len(sub) < sub_control_count - 1:
        k = rng.integers(0, 4, sample_count)
        sub += [(k == j).astype(float) for j in range(4)]
    sub_incidence = np.array(sub[: sub_control_count - 1] + [np.zeros(sample_count)]).T
    parent = []
    while len(parent) < parent_control_count:
        parent.append(rng.poisson(0.8, sample_count).astype(float))
    if total_hh_parent_control_index > 0:
        parent[total_hh_parent_control_index] = np.ones(sample_count)
    parent_incidence = np.array(parent).T

    sub_float_weights = rng.gamma(0.5, 1.0, (sub_zone_count, sample_count))
    sub_float_weights = np.rint(sub_float_weights / 1e-6) * 1e-6
    whole = rng.random(sub_float_weights.shape) < zero_resid_share
    sub_float_weights[whole] = np.floor(sub_float_weights[whole])
    sub_int_weights = sub_float_weights.astype(int)
    sub_resid_weights = sub_float_weights % 1.0
    t = 0

    relaxed = sub_float_weights @ sub_incidence
    lp_right_hand_side = np.maximum(
        np.round(relaxed) - sub_int_weights @ sub_incidence, 0.0
    )
    sub_num_households = relaxed[:, (t,)]
    sub_max = np.amax(sub_incidence, axis=0) * sub_num_households
    relax_ge_upper_bound = np.maximum(sub_max - lp_right_hand_side, 0)
    hh_constraint_ge_bound = np.maximum(sub_max, lp_right_hand_side)
    total_hh_right_hand_side = lp_right_hand_side[:, t]

    parent_float_weights = sub_float_weights.sum(axis=0)
    parent_int_weights = sub_int_weights.sum(axis=0)
    parent_resid_weights = sub_resid_weights.sum(axis=0)
    parent_lp_right_hand_side = np.maximum(
        np.round(parent_float_weights @ parent_incidence)
        - parent_int_weights @ parent_incidence,
        0.0,
    )
    parent_max = np.amax(parent_incidence, axis=0) * sub_num_households.sum()

    def importance(n):
        return rng.choice([0.0, 10.0, 100.0, 1000.0], n).astype(objective_dtype)

    return (
        sub_int_weights,
        importance(parent_control_count),
        np.maximum(parent_max - parent_lp_right_hand_side, 0),
        importance(sub_control_count),
        sub_float_weights,
        sub_resid_weights.astype(objective_dtype),
        lp_right_hand_side,
        np.maximum(parent_max, parent_lp_right_hand_side),
        sub_incidence,
        parent_incidence,
        total_hh_right_hand_side,
        relax_ge_upper_bound,
        parent_lp_right_hand_side,
        hh_constraint_ge_bound,
        parent_resid_weights.astype(objective_dtype),
        t,
        total_hh_parent_control_index,
        60,
    )


@contextlib.contextmanager
def capture_solves(export=True):
    """
    Patch pywraplp.Solver.Solve to record, per call, the deterministic
    serialization of ExportModelToProto (if export) and the time Solve
    started and took.
    """
    from ortools.linear_solver import linear_solver_pb2, pywraplp

    records = []
    original = pywraplp.Solver.Solve

    def solve(self, *args):
        record = {}
        if export:
            model = linear_solver_pb2.MPModelProto()
            self.ExportModelToProto(model)
            record["model"] = model.SerializeToString(deterministic=True)
        record["solve_start"] = time.perf_counter()
        status = original(self, *args)
        record["solve_seconds"] = time.perf_counter() - record["solve_start"]
        records.append(record)
        return status

    pywraplp.Solver.Solve = solve
    try:
        yield records
    finally:
        pywraplp.Solver.Solve = original
