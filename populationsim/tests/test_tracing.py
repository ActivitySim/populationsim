
# ActivitySim
# See full license in LICENSE.txt.

import os.path
import logging

import pytest

import pandas as pd

import activitysim.core.tracing as tracing

from activitysim.core import inject


def add_canonical_dirs():

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    inject.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    inject.add_injectable("output_dir", output_dir)


def test_config_logger(capsys):

    add_canonical_dirs()

    tracing.config_logger()

    logger = logging.getLogger('popsim')

    file_handlers = [h for h in logger.handlers if type(h) is logging.FileHandler]
    assert len(file_handlers) == 1
    asim_logger_baseFilename = file_handlers[0].baseFilename

    print("handlers:", logger.handlers)

    logger.info('test_config_logger')
    logger.info('log_info')
    logger.warning('log_warn1')

    out, err = capsys.readouterr()

    # don't consume output
    print(out)

    assert "could not find conf file" not in out
    assert 'log_warn1' in out
    assert 'log_info' not in out

    with open(asim_logger_baseFilename, 'r') as content_file:
        content = content_file.read()
        print(content)
    assert 'log_warn1' in content
    assert 'log_info' in content
