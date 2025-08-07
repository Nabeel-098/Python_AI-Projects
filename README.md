Machine Learning Mini Projects
A collection of four machine learning demos using Python, scikit-learn, XGBoost, and Gradio:

Credit Card Fraud Detection

Student Exam Result Predictor

Iris Flower Classifier

Used Car Price Predictor

Project 1: Credit Card Fraud Detection
Detect fraudulent credit card transactions using XGBoost and feature selection.

Features
Reads real dataset (creditcard.csv)

Handles data imbalance using scale_pos_weight

Top-10 feature selection (SelectKBest)

5-fold cross-validation and hyperparameter tuning

Prints ROC-AUC score

Saves the trained model as a .pkl file

Usage
Place creditcard.csv in your working directory.

Run the Python script:

text
python creditcard_fraud.py
Key Code Highlights
python
selector = SelectKBest(score_func=f_classif, k=10)
x_train_new = selector.fit_transform(x_train, y_train)
model = xgb.XGBClassifier(scale_pos_weight=weight)
grid_search = GridSearchCV(model, param_grid=params, cv=3)
Project 2: Student Exam Result Predictor
Predict if a student will pass or fail based on academic data. Uses Gradio GUI for easy input.

Features
Uses a small in-memory dataset

Feature selection via SelectKBest

Gradient Boosting Classifier with hyperparameter tuning

Interactive Gradio interface for predictions

Usage
text
pip install gradio scikit-learn pandas
python student_pass_predictor.py
Enter the student's info and get instant "Pass"/"Fail" prediction with confidence!

Key Code Highlights
python
def is_pass(hours_studied, attendance, past_grade, assignments_completed):
    ...
    prediction = best_model.predict(input_selected)[0]
    prob = best_model.predict_proba(input_selected)[0]
    return f"{label_map[prediction]} (Confidence: {max(prob)*100:.2f}%)"
Project 3: Iris Flower Classifier
Classifies iris species from user inputs using a Random Forest and Gradio web UI.

Features
Uses scikit-learn's built-in Iris dataset

Selects top-2 features with SelectKBest

Random Forest with hyperparameter tuning

Interactive web UI for feature input and prediction display

Usage
text
pip install pandas gradio scikit-learn
python iris_flower_classifier.py
Enter sepal/petal measurements and see species prediction instantly.

Key Code Highlights
python
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = best_model.predict(input_new)[0]
...
gr.Interface(fn=predict_species, inputs=[...], outputs=...)
Project 4: Used Car Price Predictor
Predicts the resale value of a used car based on its features. Gradio-based GUI.

Features
Small in-memory dataset

Encodes categorical features (brand, fuel type, transmission)

Feature engineering: computes car age

Standardizes inputs (StandardScaler)

RandomForestRegressor + GridSearchCV

Gradio UI to estimate car price

Usage
text
pip install pandas gradio scikit-learn
python used_car_price_predictor.py
Select options and enter values to get an estimated price in lakhs.

Key Code Highlights
python
input_df = pd.DataFrame([[...]], columns=[...])
input_scaled = scaler.transform(input_df)
prediction = best_model.predict(input_selected)[0]
return f"Estimated Price: ₹{prediction:.2f} lakhs"
Setup & Running
Install dependencies
Most scripts require:
pandas, scikit-learn, xgboost (project 1), and gradio (projects 2-4).

Install via:

text
pip install pandas scikit-learn xgboost gradio joblib
Run each project
Each script is self-contained and can be run directly as shown above.

For Gradio apps
Once running, a local web server will start in your browser for interactive prediction.

Notes
For Project 1, make sure to have creditcard.csv in the same directory.

All Gradio apps are for demonstration; data is for educational use.
