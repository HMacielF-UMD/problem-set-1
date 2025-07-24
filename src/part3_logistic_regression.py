'''
PART 3: Logistic Regression
- 1. Read in `df_arrests`
- 2. Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- 3. Create a list called `features` which contains our two feature names: num_fel_arrests_last_year, current_charge_felony
- 4. Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- 5. Initialize the Logistic Regression model with a variable called `lr_model` 
- 6. Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- 7. Run the model 
- 8. What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- 9. Now predict for the test set. Name this column `pred_lr`
- 10. Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr


def run_logistic_regression():
    ''' Run logistic regression on the processed arrest data
    This function reads in the preprocessed arrest data, splits it into training and test sets,
    performs hyperparameter tuning using grid search, and fits a logistic regression model.

    Parameters:
        None
    
    Returns:
        pd.DataFrame: A DataFrame containing the test set with predictions.
    '''
    
    # 1. Load the preprocessed arrest data
    df_arrests = pd.read_csv('data/df_arrests.csv')

    # 2. Split the data into training and test sets
    df_arrests_train, df_arrests_test = train_test_split(
        df_arrests, 
        test_size=0.3, 
        shuffle=True, 
        stratify=df_arrests['y'], 
        random_state=42
    )

    # 3. Define features
    features = ['num_fel_arrests_last_year', 'current_charge_felony']

    # Define Target variable
    X_train = df_arrests_train[features]
    y_train = df_arrests_train['y']
    X_test = df_arrests_test[features]

    # 4. Define parameter grid for hyperparameter tuning
    param_grid = {'C': [0.01, 0.1, 1.0]}

    # 5. Initialize the logistic regression model
    lr_model = lr(solver='liblinear')

    # 6. Initialize GridSearchCV with 5-fold cross-validation
    gs_cv = GridSearchCV(lr_model, param_grid, cv=KFold_strat(n_splits=5), scoring='accuracy')

    # 7. Fit the model to the training data
    gs_cv.fit(X_train, y_train)

    # Optimal value for C
    optimal_C = gs_cv.best_params_['C']
    
    # 8. Print optimal C and its regularization effect
    if optimal_C < 0.1:
        reg_effect = "most regularization"
    elif optimal_C > 1.0:
        reg_effect = "least regularization"
    else:
        reg_effect = "in the middle"
    print(f"Optimal value for C: {optimal_C}, which has {reg_effect}.")

    # 9. Predict on the test set
    df_arrests_test['pred_lr'] = gs_cv.predict(X_test)

    # 10. Save predictions to a CSV file for further use
    df_arrests_test.to_csv('data/df_arrests_test_with_predictions.csv', index=False)

# Run the logistic regression function if this script is executed directly
if __name__ == "__main__":
    df = run_logistic_regression()
    print("Logistic regression completed and predictions saved to 'data/df_arrests_test_with_predictions.csv'.")

