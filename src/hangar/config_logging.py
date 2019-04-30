import logging
import logging.config
import logging.handlers
import os

import yaml


def setup_logging(cfg_path='config_logging.yml', cfg_level=logging.INFO, env_key='HANGAR_LOG_CFG'):
    '''setup logging configuration for a hangar project

    the location of the log file can be modified by setting path or the
    `HANGAR_LOG_CFG` environment key.

    Parameters
    ----------
    cfg_path : str, optional
        path to a yaml file on disk containing logging configuration. (the
        default is 'logging.yaml')
    cfg_level : logging.LEVEL, optional
        Level to log messages at (the default is logging.INFO, which is for
        normal use)
    env_key : str, optional
        environment key whcih can be used to set a path to the logging config
        file. (the default is 'HANGAR_LOG_CFG')

    '''
    path = os.path.join(os.path.dirname(__file__), cfg_path)
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=cfg_level)
