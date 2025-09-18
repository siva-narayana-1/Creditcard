<<<<<<< HEAD
import sys
import logging

class Logger:
    def get_logs(log_name):
        try:
            logger = logging.getLogger(log_name)
            logger.setLevel(logging.DEBUG)

            handler = logging.FileHandler(f'C:\\project\\Logs\\{log_name}.log')
            formate = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            handler.setFormatter(formate)
=======
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
>>>>>>> b1bbf0c870a1d8ca10d3460eb94dab38a20b0b6a
            logger.addHandler(handler)

            return logger
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
<<<<<<< HEAD
            print(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')
=======
            print(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")


>>>>>>> b1bbf0c870a1d8ca10d3460eb94dab38a20b0b6a
