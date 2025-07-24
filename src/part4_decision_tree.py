'''
PART 4: Decision Trees
- 1. Read in the dataframe(s) from PART 3
- 2. Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- 3. Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- 4. Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. 
Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- 5. Run the model 
- 6. What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- 7. Now predict for the test set. Name this column `pred_dt` 
- 8. Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC



def run_decision_tree():
    ''' 
    Run the Decision Tree model and save results

    Parameters:
        None
        
    Returns:
        None
    '''
    
    # 1. Read in the dataframe
    df = pd.read_csv('data/part3_data.csv')  # Adjust the path if needed

    # 2. Create a parameter grid for Decision Tree
    param_grid_dt = { 'max_depth': [1, 5, 10] }

    # 3. Initialize the Decision Tree model
    dt_model = DTC(random_state=42)

    # 4. Set up Stratified K-Fold cross-validation
    cv = KFold_strat(n_splits=5, shuffle=True, random_state=42)
    gs_cv_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=cv)

    # Prepare features and target
    X = df.drop(columns=['arrested', 'predicted_risk'])  # Features
    y = df['arrested']  # Target variable   
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Run the model
    gs_cv_dt.fit(X_train, y_train)

    # 6. Optimal value for max_depth
    optimal_max_depth = gs_cv_dt.best_params_['max_depth']
    print(f"Optimal max_depth: {optimal_max_depth}")
    if optimal_max_depth == 1:
        print("This has the most regularization.")  
    elif optimal_max_depth == 10:
        print("This has the least regularization.")
    else:
        print("This is in the middle of regularization.")
    
    # 7. Predict for the test set
    X_test = X_test.copy()  # Avoid SettingWithCopyWarning
    y_pred_dt = gs_cv_dt.predict(X_test)
    X_test['pred_dt'] = y_pred_dt
    X_test['predicted_risk'] = gs_cv_dt.predict_proba(X_test)[:, 1] 
    X_test['arrested'] = y_test.values 

    # 8. Save the results to a CSV file for PART 5
    X_test.to_csv('data/part4_decision_tree_results.csv', index=False)