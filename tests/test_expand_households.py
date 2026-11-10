import numpy as np
import pandas as pd

from populationsim.steps.expand_households import choose_group_households

HH_ID_COL = "hh_id"


def all_groups_choices(group_ids, household_groups, prng):
    # the pre-change algorithm: probability lists built for every group
    grouper = household_groups.groupby("group_id")
    group_hh_probs = [0] * len(grouper)
    for group_id, df in grouper:
        hh_ids = list(df[HH_ID_COL])
        probs = list(df.sample_weight / df.sample_weight.sum())
        group_hh_probs[group_id] = [hh_ids, probs]

    def chooser(group_id):
        return prng.choice(group_hh_probs[group_id][0], p=group_hh_probs[group_id][1])

    return group_ids.apply(chooser)


def synthetic_inputs(n_groups, n_used, seed):
    rng = np.random.RandomState(seed)
    group_size = rng.randint(1, 6, size=n_groups)
    group_id = np.repeat(np.arange(n_groups), group_size)
    household_groups = pd.DataFrame(
        {
            HH_ID_COL: rng.permutation(len(group_id)) + 1000,
            "group_id": group_id,
            "sample_weight": rng.uniform(0.5, 50.0, size=len(group_id)),
        }
    )
    # seed households are not stored in group order
    household_groups = household_groups.sample(frac=1.0, random_state=rng)

    used = rng.choice(n_groups, size=n_used, replace=False)
    draws = rng.randint(1, 20, size=n_used)
    group_ids = pd.Series(np.sort(np.repeat(used, draws)), name="group_id")
    return household_groups, group_ids


def test_choices_match_all_groups_algorithm():
    household_groups, group_ids = synthetic_inputs(n_groups=5000, n_used=40, seed=7)
    assert group_ids.nunique() < household_groups.group_id.nunique() / 100

    expected = all_groups_choices(group_ids, household_groups, np.random.RandomState(1))
    chosen = choose_group_households(
        group_ids, household_groups, HH_ID_COL, np.random.RandomState(1)
    )

    assert chosen.index.equals(group_ids.index)
    assert chosen.tolist() == expected.tolist()


def test_choices_match_when_every_group_is_used():
    household_groups, group_ids = synthetic_inputs(n_groups=200, n_used=200, seed=11)

    expected = all_groups_choices(group_ids, household_groups, np.random.RandomState(3))
    chosen = choose_group_households(
        group_ids, household_groups, HH_ID_COL, np.random.RandomState(3)
    )

    assert chosen.tolist() == expected.tolist()


def test_chosen_households_belong_to_their_group():
    household_groups, group_ids = synthetic_inputs(n_groups=1000, n_used=25, seed=5)

    chosen = choose_group_households(
        group_ids, household_groups, HH_ID_COL, np.random.RandomState(0)
    )

    hh_group = household_groups.set_index(HH_ID_COL).group_id
    assert (hh_group.loc[chosen.values].values == group_ids.values).all()
