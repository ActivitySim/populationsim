# ActivitySim
# See full license in LICENSE.txt.
import importlib
import logging
import os
import sys
import warnings
import contextlib
import io
import numpy as np
import time
from datetime import timedelta

from populationsim.core import config, inject, mem, pipeline, tracing

logger = logging.getLogger(__name__)


INJECTABLES = [
    "data_dir",
    "configs_dir",
    "output_dir",
    "settings_file_name",
    "imported_extensions",
]


class TimeLogger:

    aggregate_timing = {}

    def __init__(self, tag1):
        self._time_point = self._time_start = time.time()
        self._time_log = []
        self._tag1 = tag1

    def summary(self, logger, tag, level=20, suffix=None):
        gross_elaspsed = time.time() - self._time_start
        if suffix:
            msg = f"{tag} in {timedelta(seconds=gross_elaspsed)}: ({suffix})\n"
        else:
            msg = f"{tag} in {timedelta(seconds=gross_elaspsed)}: \n"
        msgs = []
        for i in self._time_log:
            j = timedelta(seconds=self.aggregate_timing[f"{self._tag1}.{i[0]}"])
            msgs.append("   - {0:24s} {1} [{2}]".format(*i, j))
        msg += "\n".join(msgs)
        logger.log(level=level, msg=msg)

    @classmethod
    def aggregate_summary(
        cls, logger, heading="Aggregate Flow Timing Summary", level=20
    ):
        msg = f"{heading}\n"
        msgs = []
        for tag, elapsed in cls.aggregate_timing.items():
            msgs.append("   - {0:48s} {1}".format(tag, timedelta(seconds=elapsed)))
        msg += "\n".join(msgs)
        logger.log(level=level, msg=msg)


def add_run_args(parser, multiprocess=True):
    """Run command args"""
    parser.add_argument(
        "-w",
        "--working_dir",
        type=str,
        metavar="PATH",
        help="path to example/project directory (default: %s)" % os.getcwd(),
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        action="append",
        metavar="PATH",
        help="path to config dir",
    )
    parser.add_argument(
        "-o", "--output", type=str, metavar="PATH", help="path to output dir"
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        action="append",
        metavar="PATH",
        help="path to data dir",
    )
    parser.add_argument(
        "-r", "--resume", type=str, metavar="STEPNAME", help="resume after step"
    )
    parser.add_argument(
        "-p", "--pipeline", type=str, metavar="FILE", help="pipeline file name"
    )
    parser.add_argument(
        "-s", "--settings_file", type=str, metavar="FILE", help="settings file name"
    )
    parser.add_argument(
        "--households_sample_size", type=int, metavar="N", help="households sample size"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Do not limit process to one thread. "
        "Can make single process runs faster, "
        "but will cause thrashing on MP runs.",
    )
    parser.add_argument(
        "-e",
        "--ext",
        type=str,
        action="append",
        metavar="PATH",
        help="Package of extension modules to load. Use of this option is not "
        "generally secure.",
    )

    if multiprocess:
        parser.add_argument(
            "-m",
            "--multiprocess",
            default=False,
            const=-1,
            metavar="(N)",
            nargs="?",
            type=int,
            help="run multiprocess. Adds configs_mp settings"
            " by default. Optionally give a number of processes,"
            " which will override the settings file.",
        )


def validate_injectable(name):
    try:
        dir_paths = inject.get_injectable(name)
    except RuntimeError:
        # injectable is missing, meaning is hasn't been explicitly set
        # and defaults cannot be found.
        sys.exit(
            f"Error({name}): please specify either a --working_dir "
            "containing 'configs', 'data', and 'output' folders "
            "or all three of --config, --data, and --output"
        )

    dir_paths = [dir_paths] if isinstance(dir_paths, str) else dir_paths

    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            sys.exit("Could not find %s '%s'" % (name, os.path.abspath(dir_path)))

    return dir_paths


def handle_standard_args(args, multiprocess=True):
    def inject_arg(name, value, cache=False):
        assert name in INJECTABLES
        inject.add_injectable(name, value, cache=cache)

    if args.working_dir:
        # populationsim will look in the current working directory for
        # 'configs', 'data', and 'output' folders by default
        os.chdir(args.working_dir)

    if args.ext:
        for e in args.ext:
            basepath, extpath = os.path.split(e)
            if not basepath:
                basepath = "."
            sys.path.insert(0, os.path.abspath(basepath))
            try:
                importlib.import_module(extpath)
            except ImportError as err:
                logger.exception(f"ImportError {err}")
                raise
            except Exception as err:
                logger.exception(f"Error {err}")
                raise
            finally:
                del sys.path[0]
        inject_arg("imported_extensions", args.ext)
    else:
        inject_arg("imported_extensions", ())

    # settings_file_name should be cached or else it gets squashed by config.py
    if args.settings_file:
        inject_arg("settings_file_name", args.settings_file, cache=True)

    if args.config:
        inject_arg("configs_dir", args.config)

    if args.data:
        inject_arg("data_dir", args.data)

    if args.output:
        inject_arg("output_dir", args.output)

    if multiprocess and args.multiprocess:
        config_paths = validate_injectable("configs_dir")

        if not os.path.exists("configs_mp"):
            logger.warning("could not find 'configs_mp'. skipping...")
        else:
            logger.info("adding 'configs_mp' to config_dir list...")
            config_paths.insert(0, "configs_mp")
            inject_arg("configs_dir", config_paths)

        config.override_setting("multiprocess", True)
        if args.multiprocess > 0:
            config.override_setting("num_processes", args.multiprocess)

    if args.households_sample_size is not None:
        config.override_setting("households_sample_size", args.households_sample_size)

    for injectable in ["configs_dir", "data_dir", "output_dir"]:
        validate_injectable(injectable)

    if args.pipeline:
        inject.add_injectable("pipeline_file_name", args.pipeline)

    if args.resume:
        config.override_setting("resume_after", args.resume)


def cleanup_output_files():

    tracing.delete_trace_files()

    csv_ignore = []
    if config.setting("memory_profile", False):
        # memory profiling is opened potentially before `cleanup_output_files`
        # is called, but we want to leave any (newly created) memory profiling
        # log files that may have just been created.
        mem_prof_log = config.log_file_path("memory_profile.csv")
        csv_ignore.append(mem_prof_log)

    tracing.delete_output_files("h5")
    tracing.delete_output_files("csv", ignore=csv_ignore)
    tracing.delete_output_files("txt")
    tracing.delete_output_files("yaml")
    tracing.delete_output_files("prof")
    tracing.delete_output_files("omx")


def run(args):
    """
    Run the models. Specify a project folder using the '--working_dir' option,
    or point to the config, data, and output folders directly with
    '--config', '--data', and '--output'. Both '--config' and '--data' can be
    specified multiple times. Directories listed first take precedence.

    returns:
        int: sys.exit exit code
    """

    # register steps and other injectables
    if not inject.is_injectable("preload_injectables"):
        pass

    tracing.config_logger(basic=True)
    handle_standard_args(args)  # possibly update injectables

    if config.setting("rotate_logs", False):
        config.rotate_log_directory()

    if config.setting("memory_profile", False) and not config.setting(
        "multiprocess", False
    ):
        # Memory sidecar is only useful for single process runs
        # multiprocess runs log memory usage without blocking in the controlling process.
        mem_prof_log = config.log_file_path("memory_profile.csv")
        from populationsim.core.memory_sidecar import MemorySidecar

        memory_sidecar_process = MemorySidecar(mem_prof_log)
    else:
        memory_sidecar_process = None

    # legacy support for run_list setting nested 'models' and 'resume_after' settings
    if config.setting("run_list"):
        warnings.warn(
            "Support for 'run_list' settings group will be removed.\n"
            "The run_list.steps setting is renamed 'models'.\n"
            "The run_list.resume_after setting is renamed 'resume_after'.\n"
            "Specify both 'models' and 'resume_after' directly in settings config file.",
            FutureWarning,
            stacklevel=2,
        )
        run_list = config.setting("run_list")
        if "steps" in run_list:
            assert not config.setting(
                "models"
            ), "Don't expect 'steps' in run_list and 'models' as stand-alone setting!"
            config.override_setting("models", run_list["steps"])

        if "resume_after" in run_list:
            assert not config.setting(
                "resume_after"
            ), "Don't expect 'resume_after' both in run_list and as stand-alone setting!"
            config.override_setting("resume_after", run_list["resume_after"])

    # If you provide a resume_after argument to pipeline.run
    # the pipeline manager will attempt to load checkpointed tables from the checkpoint store
    # and resume pipeline processing on the next submodel step after the specified checkpoint
    resume_after = config.setting("resume_after", None)

    # cleanup if not resuming
    if not resume_after:
        cleanup_output_files()
    elif config.setting("cleanup_trace_files_on_resume", False):
        tracing.delete_trace_files()

    tracing.config_logger(basic=False)  # update using possibly new logging configs
    config.filter_warnings()
    logging.captureWarnings(capture=True)

    # directories
    for k in ["configs_dir", "settings_file_name", "data_dir", "output_dir"]:
        logger.info("SETTING %s: %s" % (k, inject.get_injectable(k, None)))

    log_settings = inject.get_injectable("log_settings", {})
    for k in log_settings:
        logger.info("SETTING %s: %s" % (k, config.setting(k)))

    # OMP_NUM_THREADS: openmp
    # OPENBLAS_NUM_THREADS: openblas
    # MKL_NUM_THREADS: mkl
    for env in [
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ]:
        logger.info(f"ENV {env}: {os.getenv(env)}")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    logger.info("NumPy build info:\n%s", buf.getvalue())

    t0 = tracing.print_elapsed_time()

    try:
        if config.setting("multiprocess", False):
            logger.info("run multiprocess simulation")

            from populationsim.core import mp_tasks

            injectables = {k: inject.get_injectable(k) for k in INJECTABLES}
            mp_tasks.run_multiprocess(injectables)

            assert not pipeline.is_open()

            if config.setting("cleanup_pipeline_after_run", False):
                pipeline.cleanup_pipeline()

        else:
            logger.info("run single process simulation")

            pipeline.run(
                models=config.setting("models"),
                resume_after=resume_after,
                memory_sidecar_process=memory_sidecar_process,
            )

            if config.setting("cleanup_pipeline_after_run", False):
                pipeline.cleanup_pipeline()  # has side effect of closing open pipeline
            else:
                pipeline.close_pipeline()

            mem.log_global_hwm()  # main process
    except Exception:
        # log time until error and the error traceback
        tracing.print_elapsed_time("all models until this error", t0)
        logger.exception("populationsim run encountered an unrecoverable error")
        raise

    mem.consolidate_logs()

    TimeLogger.aggregate_summary(logger)

    tracing.print_elapsed_time("all models", t0)

    if memory_sidecar_process:
        memory_sidecar_process.stop()

    return 0
