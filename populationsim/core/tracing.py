# ActivitySim
# See full license in LICENSE.txt.

import logging
import logging.config
import multiprocessing  # for process name
import os
import sys
import time
import yaml

from populationsim.core import inject, config

# Configurations
ASIM_LOGGER = "populationsim"
CSV_FILE_TYPE = "csv"
LOGGING_CONF_FILE_NAME = "logging.yaml"


logger = logging.getLogger(__name__)

timing_notes = set()


class ElapsedTimeFormatter(logging.Formatter):
    def format(self, record):
        duration_milliseconds = record.relativeCreated
        hours, rem = divmod(duration_milliseconds / 1000, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            record.elapsedTime = "{:0>2}:{:0>2}:{:05.2f}".format(
                int(hours), int(minutes), seconds
            )
        else:
            record.elapsedTime = "{:0>2}:{:05.2f}".format(int(minutes), seconds)
        return super(ElapsedTimeFormatter, self).format(record)


def format_elapsed_time(t):
    return "%s seconds (%s minutes)" % (round(t, 3), round(t / 60.0, 1))


def print_elapsed_time(msg=None, t0=None, debug=False):
    t1 = time.time()
    if msg:
        assert t0 is not None
        t = t1 - (t0 or t1)
        msg = "Time to execute %s : %s" % (msg, format_elapsed_time(t))
        if debug:
            logger.debug(msg)
        else:
            logger.info(msg)
    return t1


def log_runtime(model_name, start_time=None, timing=None, force=False):
    global timing_notes

    assert (start_time or timing) and not (start_time and timing)

    timing = timing if timing else time.time() - start_time
    seconds = round(timing, 1)
    minutes = round(timing / 60, 1)

    process_name = multiprocessing.current_process().name

    if config.setting("multiprocess", False) and not force:
        # when benchmarking, log timing for each processes in its own log
        if config.setting("benchmarking", False):
            header = "component_name,duration"
            with config.open_log_file(
                f"timing_log.{process_name}.csv", "a", header
            ) as log_file:
                print(f"{model_name},{timing}", file=log_file)
        # only continue to log runtime in global timing log for locutor
        if not inject.get_injectable("locutor", False):
            return

    header = "process_name,model_name,seconds,minutes,notes"
    note = " ".join(timing_notes)
    with config.open_log_file("timing_log.csv", "a", header) as log_file:
        print(f"{process_name},{model_name},{seconds},{minutes},{note}", file=log_file)

    timing_notes.clear()


def delete_output_files(file_type, ignore=None, subdir=None):
    """
    Delete files in output directory of specified type

    Parameters
    ----------
    output_dir: str
        Directory of trace output CSVs

    Returns
    -------
    Nothing
    """

    output_dir = inject.get_injectable("output_dir")

    subdir = [subdir] if subdir else None
    directories = subdir or ["", "log", "trace"]

    for subdir in directories:

        dir = os.path.join(output_dir, subdir) if subdir else output_dir

        if not os.path.exists(dir):
            continue

        if ignore:
            ignore = [os.path.realpath(p) for p in ignore]

        # logger.debug("Deleting %s files in output dir %s" % (file_type, dir))

        for the_file in os.listdir(dir):
            if the_file.endswith(file_type):
                file_path = os.path.join(dir, the_file)

                if ignore and os.path.realpath(file_path) in ignore:
                    continue

                try:
                    if os.path.isfile(file_path):
                        logger.debug("delete_output_files deleting %s" % file_path)
                        os.unlink(file_path)
                except Exception as e:
                    print(e)


def delete_trace_files():
    """
    Delete CSV files in output_dir

    Returns
    -------
    Nothing
    """
    delete_output_files(CSV_FILE_TYPE, subdir="trace")
    delete_output_files(CSV_FILE_TYPE, subdir="log")

    active_log_files = [
        h.baseFilename
        for h in logger.root.handlers
        if isinstance(h, logging.FileHandler)
    ]

    delete_output_files("log", ignore=active_log_files)


def config_logger(basic=False):
    """
    Configure logger

    look for conf file in configs_dir, if not found use basicConfig

    Returns
    -------
    Nothing
    """

    # look for conf file in configs_dir
    if basic:
        log_config_file = None
    else:
        log_config_file = config.config_file_path(
            LOGGING_CONF_FILE_NAME, mandatory=False
        )

    if log_config_file:
        try:
            with open(log_config_file) as f:
                config_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
        except Exception as e:
            print(f"Unable to read logging config file {log_config_file}")
            raise e

        try:
            config_dict = config_dict["logging"]
            config_dict.setdefault("version", 1)
            logging.config.dictConfig(config_dict)
        except Exception as e:
            print(f"Unable to config logging as specified in {log_config_file}")
            raise e

    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    logger = logging.getLogger(ASIM_LOGGER)

    if log_config_file:
        logger.info("Read logging configuration from: %s" % log_config_file)
    else:
        print("Configured logging using basicConfig")
        logger.info("Configured logging using basicConfig")


def initialize_traceable_tables():

    traceable_table_ids = inject.get_injectable("traceable_table_ids", {})
    if len(traceable_table_ids) > 0:
        logger.debug(
            f"initialize_traceable_tables resetting table_ids for {list(traceable_table_ids.keys())}"
        )
    inject.add_injectable("traceable_table_ids", {})


def register_traceable_table(table_name, df):
    """
    Register traceable table

    Parameters
    ----------
    df: pandas.DataFrame
        traced dataframe

    Returns
    -------
    Nothing
    """

    # add index name to traceable_table_indexes

    logger.debug(f"register_traceable_table {table_name}")

    traceable_tables = inject.get_injectable("traceable_tables", [])
    if table_name not in traceable_tables:
        logger.error("table '%s' not in traceable_tables" % table_name)
        return

    idx_name = df.index.name
    if idx_name is None:
        logger.error("Can't register table '%s' without index name" % table_name)
        return

    traceable_table_ids = inject.get_injectable("traceable_table_ids", {})
    traceable_table_indexes = inject.get_injectable("traceable_table_indexes", {})

    if (
        idx_name in traceable_table_indexes
        and traceable_table_indexes[idx_name] != table_name
    ):
        logger.error(
            "table '%s' index name '%s' already registered for table '%s'"
            % (table_name, idx_name, traceable_table_indexes[idx_name])
        )
        return

    # update traceable_table_indexes with this traceable_table's idx_name
    if idx_name not in traceable_table_indexes:
        traceable_table_indexes[idx_name] = table_name
        logger.debug(
            "adding table %s.%s to traceable_table_indexes" % (table_name, idx_name)
        )
        inject.add_injectable("traceable_table_indexes", traceable_table_indexes)

    # add any new indexes associated with trace_hh_id to traceable_table_ids

    trace_hh_id = inject.get_injectable("trace_hh_id", None)
    if trace_hh_id is None:
        return

    new_traced_ids = []
    # if table_name == "households":
    if table_name in ["households", "proto_households"]:
        if trace_hh_id not in df.index:
            logger.warning("trace_hh_id %s not in dataframe" % trace_hh_id)
            new_traced_ids = []
        else:
            logger.info(
                "tracing household id %s in %s households"
                % (trace_hh_id, len(df.index))
            )
            new_traced_ids = [trace_hh_id]
    else:

        # find first already registered ref_col we can use to slice this table
        ref_col = next((c for c in traceable_table_indexes if c in df.columns), None)

        if ref_col is None:
            logger.error(
                "can't find a registered table to slice table '%s' index name '%s'"
                " in traceable_table_indexes: %s"
                % (table_name, idx_name, traceable_table_indexes)
            )
            return

        # get traceable_ids for ref_col table
        ref_col_table_name = traceable_table_indexes[ref_col]
        ref_col_traced_ids = traceable_table_ids.get(ref_col_table_name, [])

        # inject list of ids in table we are tracing
        # this allows us to slice by id without requiring presence of a household id column
        traced_df = df[df[ref_col].isin(ref_col_traced_ids)]
        new_traced_ids = traced_df.index.tolist()
        if len(new_traced_ids) == 0:
            logger.warning(
                "register %s: no rows with %s in %s."
                % (table_name, ref_col, ref_col_traced_ids)
            )

    # update the list of trace_ids for this table
    prior_traced_ids = traceable_table_ids.get(table_name, [])

    if new_traced_ids:
        assert not set(prior_traced_ids) & set(new_traced_ids)
        traceable_table_ids[table_name] = prior_traced_ids + new_traced_ids
        inject.add_injectable("traceable_table_ids", traceable_table_ids)

    logger.debug(
        "register %s: added %s new ids to %s existing trace ids"
        % (table_name, len(new_traced_ids), len(prior_traced_ids))
    )
    logger.debug(
        "register %s: tracing new ids %s in %s"
        % (table_name, new_traced_ids, table_name)
    )
