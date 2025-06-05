import logging

import colorlog


def setup_logging(level=logging.INFO):
    if isinstance(level, str):
        level_dict = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'error': logging.ERROR
        }
        level = level_dict.get(level.lower(), logging.INFO)

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers = [handler]

    return logger
