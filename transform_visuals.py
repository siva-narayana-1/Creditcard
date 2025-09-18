import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy import stats
from loggers import Logs
logger = Logs.get_logger('visual_logs')
class Visuals:
    def visual_outliers(x_train, var, x_test):
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows: train & test, 3 plots each

            # ---------- Train Visuals ----------
            axes[0, 0].set_title(f"{var} Train Normal Distribution")
            x_train[var].plot(kind="kde", color='r', ax=axes[0, 0])

            axes[0, 1].set_title(f"{var} Train Outliers")
            sns.boxplot(x=x_train[var], ax=axes[0, 1])

            axes[0, 2].set_title(f"{var} Train Q-Q Plot")
            stats.probplot(x_train[var], dist="norm", plot=axes[0, 2])

            # ---------- Test Visuals ----------
            axes[1, 0].set_title(f"{var} Test Normal Distribution")
            x_test[var].plot(kind="kde", color='b', ax=axes[1, 0])

            axes[1, 1].set_title(f"{var} Test Outliers")
            sns.boxplot(x=x_test[var], ax=axes[1, 1])

            axes[1, 2].set_title(f"{var} Test Q-Q Plot")
            stats.probplot(x_test[var], dist="norm", plot=axes[1, 2])

            plt.tight_layout()
            logger.info(f"Successfully created visuals.")
            # plt.show()

        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at line {exc_line.tb_lineno} → {exc_msg}")
