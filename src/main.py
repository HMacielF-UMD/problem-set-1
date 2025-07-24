'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import part1_etl as etl
import part2_preprocessing as preprocessing
import part3_logistic_regression as logistic_regression
import part4_decision_tree as decision_tree
import part5_calibration_plot as calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    etl.run_etl()

    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests = preprocessing.run_preprocessing()

    # PART 3: Call functions/instanciate objects from logistic_regression
    df_logistic = logistic_regression.run_logistic_regression()

    # PART 4: Call functions/instanciate objects from decision_tree
    df_decision_tree = decision_tree.run_decision_tree()

    # PART 5: Call functions/instanciate objects from calibration_plot
    calibration_plot.calibration_plot(df_logistic['y'].to_list(), df_decision_tree['pred_dt'].to_list(), n_bins=5)

# Print final confirmation
if __name__ == "__main__":
    main()