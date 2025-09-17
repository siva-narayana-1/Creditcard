import logging
import sys

def get_logger(name):
    try:
        logger = logging.getLogger('logs')
        logger.setLevel(logging.DEBUG)

        handlers = logging.FileHandler(f'C:\\ML_all\\Logs\\{name}.log')
        formate = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        handlers.setFormatter(formate)
        logger.addHandler(handlers)

        return logger
    except Exception as e:
        exc_type, exc_msg, exc_line = sys.exc_info()
        return f"{exc_line.tb_lineno} errror as {exc_msg}"