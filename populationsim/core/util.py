# ActivitySim
# See full license in LICENSE.txt.

import argparse
import logging
import os

logger = logging.getLogger(__name__)


def si_units(x, kind="B", digits=3, shift=1000):

    #       nano micro milli    kilo mega giga tera peta exa  zeta yotta
    tiers = ["n", "µ", "m", "", "K", "M", "G", "T", "P", "E", "Z", "Y"]

    tier = 3
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x > 0:
        while x > shift and tier < len(tiers):
            x /= shift
            tier += 1
        while x < 1 and tier >= 0:
            x *= shift
            tier -= 1
    return f"{sign}{round(x,digits)} {tiers[tier]}{kind}"


def GB(bytes):
    return si_units(bytes, kind="B", digits=1)


def SEC(seconds):
    return si_units(seconds, kind="s", digits=2)


def INT(x):
    # format int as camel case (e.g. 1000000 vecomes '1_000_000')
    negative = x < 0
    x = abs(int(x))
    result = ""
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = "_%03d%s" % (r, result)
    result = "%d%s" % (x, result)

    return f"{'-' if negative else ''}{result}"


def delete_files(file_list, trace_label):
    # delete files in file_list

    file_list = [file_list] if isinstance(file_list, str) else file_list
    for file_path in file_list:
        try:
            if os.path.isfile(file_path):
                logger.debug(f"{trace_label} deleting {file_path}")
                os.unlink(file_path)
        except Exception:
            logger.warning(f"{trace_label} exception (e) trying to delete {file_path}")


def df_size(df):
    bytes = 0 if df.empty else df.memory_usage(index=True).sum()
    return "%s %s" % (df.shape, GB(bytes))


def reindex(series1, series2):
    """
    This reindexes the first series by the second series.  This is an extremely
    common operation that does not appear to  be in Pandas at this time.
    If anyone knows of an easier way to do this in Pandas, please inform the
    UrbanSim developers.

    The canonical example would be a parcel series which has an index which is
    parcel_ids and a value which you want to fetch, let's say it's land_area.
    Another dataset, let's say of buildings has a series which indicate the
    parcel_ids that the buildings are located on, but which does not have
    land_area.  If you pass parcels.land_area as the first series and
    buildings.parcel_id as the second series, this function returns a series
    which is indexed by buildings and has land_area as values and can be
    added to the buildings dataset.

    In short, this is a join on to a different table using a foreign key
    stored in the current table, but with only one attribute rather than
    for a full dataset.

    This is very similar to the pandas "loc" function or "reindex" function,
    but neither of those functions return the series indexed on the current
    table.  In both of those cases, the series would be indexed on the foreign
    table and would require a second step to change the index.

    Parameters
    ----------
    series1, series2 : pandas.Series

    Returns
    -------
    reindexed : pandas.Series

    """

    result = series1.reindex(series2)
    try:
        result.index = series2.index
    except AttributeError:
        pass
    return result


def recursive_replace(obj, search, replace):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = recursive_replace(v, search, replace)
    if isinstance(obj, list):
        obj = [replace if x == search else x for x in obj]
    if search == obj:
        obj = replace
    return obj


def suffix_tables_in_settings(
    model_settings,
    suffix="proto_",
    tables=None,
):
    if tables is None:
        tables = [
            "persons",
            "households",
            "tours",
            "persons_merged",
        ]
    for k in tables:
        model_settings = recursive_replace(model_settings, k, suffix + k)
    return model_settings



def parse_suffix_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="file name")
    parser.add_argument("-s", "--SUFFIX", "-s", help="suffix to replace root targets")
    parser.add_argument(
        "-r", "--ROOTS", nargs="*", help="roots be suffixed", default=[]
    )
    return parser.parse_args(args.split())


def flatten(lst):
    for sublist in lst:
        if isinstance(sublist, list):
            for item in sublist:
                yield item
        else:
            yield sublist

