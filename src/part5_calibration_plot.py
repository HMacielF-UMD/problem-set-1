'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

    

def run_extra_credit():
    """
    Runs calibration and evaluation steps for both models.
    Assumes data files were created in prior steps.
    """

    # === 1. Load prediction data from Part 3 and Part 4 ===
    df_lr = pd.read_csv('data/df_arrests_test_with_predictions.csv') 
    df_dt = pd.read_csv('data/decision_tree_results.csv')

    # === 2. Calibration Plots ===
    print("Calibration Plot - Logistic Regression:")
    calibration_plot(df_lr['y'], df_lr['pred_lr'], n_bins=5)

    print("Calibration Plot - Decision Tree:")
    calibration_plot(df_dt['arrested'], df_dt['predicted_risk'], n_bins=5)

    print("Which model is more calibrated?")
    print("→ Visually inspect the plots. The model whose curve is closer to the 45-degree dashed line is more calibrated.")

    # === 3. Extra Credit Metrics ===

    # --- PPV for Top 50 ---
    top50_lr = df_lr.sort_values(by='pred_lr', ascending=False).head(50)
    ppv_lr = top50_lr['y'].mean()

    top50_dt = df_dt.sort_values(by='predicted_risk', ascending=False).head(50)
    ppv_dt = top50_dt['arrested'].mean()

    print(f"PPV (Top 50) - Logistic Regression: {ppv_lr:.2%}")
    print(f"PPV (Top 50) - Decision Tree: {ppv_dt:.2%}")

    # --- AUC Scores ---
    auc_lr = roc_auc_score(df_lr['y'], df_lr['pred_lr'])
    auc_dt = roc_auc_score(df_dt['arrested'], df_dt['predicted_risk'])

    print(f"AUC - Logistic Regression: {auc_lr:.3f}")
    print(f"AUC - Decision Tree: {auc_dt:.3f}")

    # --- Interpretation ---
    print("Do both metrics agree that one model is more accurate than the other?")
    if auc_lr > auc_dt and ppv_lr > ppv_dt:
        print("→ Yes, both AUC and PPV suggest that Logistic Regression is more accurate.")
    elif auc_lr < auc_dt and ppv_lr < ppv_dt:
        print("→ Yes, both AUC and PPV suggest that Decision Tree is more accurate.")
    else:
        print("→ No, the metrics disagree on which model is more accurate.")
