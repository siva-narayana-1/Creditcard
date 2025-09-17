import logging
import sys
from cgitb import handler

class Logs:
    def get_logger(script_name):
        try:
            logger = logging.getLogger(script_name)
            logger.setLevel(logging.DEBUG)

            # Create a file handler for the script
            handler = logging.FileHandler(f'C:\\credict_card\\logs\\{script_name}.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            return logger
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            print(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")


