import numpy as np


def smart_round(int_weights, resid_weights, target_sum):
    """
    Round weights while ensuring (as far as possible that result sums to target_sum)

    Parameters
    ----------
    int_weights : numpy.ndarray(int)
    resid_weights : numpy.ndarray(float)
    target_sum : int

    Returns
    -------
    rounded_weights : numpy.ndarray array of ints
    """
    assert len(int_weights) == len(resid_weights)
    assert (int_weights == int_weights.astype(int)).all()
    assert target_sum == int(target_sum)

    target_sum = int(target_sum)

    # integer part of numbers to round (astype both copies and coerces)
    rounded_weights = int_weights.astype(int)

    # find number of residuals that we need to round up
    int_shortfall = target_sum - rounded_weights.sum()

    # clip to feasible, in case target was not achievable by rounding
    int_shortfall = np.clip(int_shortfall, 0, len(resid_weights))

    # Order the residual weights and round at the tipping point where target_sum is achieved
    if int_shortfall > 0:
        # indices of the int_shortfall highest resid_weights
        i = np.argsort(resid_weights)[-int_shortfall:]

        # add 1 to the integer weights that we want to round upwards
        rounded_weights[i] += 1

    return rounded_weights
