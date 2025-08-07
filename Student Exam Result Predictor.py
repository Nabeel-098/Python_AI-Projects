import pandas as pd
import gradio as gr
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif

data = {
    "hours_studied": [5, 2, 8, 1, 7, 6, 3, 4, 9, 0, 2, 5, 7, 6, 3],
    "attendance": [85, 60, 95, 50, 90, 80, 70, 75, 98, 40, 65, 88, 92, 85, 68],
    "assignments_completed": [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    "past_grade": [60, 40, 75, 30, 65, 70, 45, 55, 85, 25, 38, 72, 78, 69, 50],
    "pass_or_fail": ["Pass", "Fail", "Pass", "Fail", "Pass", "Pass", "Fail", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Fail"]
}

df = pd.DataFrame(data)
df['pass_or_fail'] = df['pass_or_fail'].map({'Pass':1,'Fail':0})
label_map = {1: "Pass", 0: "Fail"}

x = df.drop(['pass_or_fail'], axis=1)
y = df['pass_or_fail']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

selector = SelectKBest(score_func=f_classif,k=4)
x_train_new = selector.fit_transform(x_train, y_train)
x_test_new = selector.transform(x_test)

model = GradientBoostingClassifier()

scores = cross_val_score(model, x_train_new, y_train, cv=5)

params = {
    'n_estimators':[10, 50],
    'max_depth':[2,4]
}
grid_search = GridSearchCV(model, param_grid=params, cv=3)
grid_search.fit(x_train_new, y_train)
best_model = grid_search.best_estimator_

def is_pass(hours_studied, attendance, past_grade, assignments_completed):
    input_data = [[hours_studied, attendance, assignments_completed, past_grade]]
    input_selected = selector.transform(input_data)
    prediction = best_model.predict(input_selected)[0]
    prob = best_model.predict_proba(input_selected)[0]
    return f"{label_map[prediction]} (Confidence: {max(prob)*100:.2f}%)"




interface = gr.Interface(
    fn=is_pass,
    inputs=[
        gr.Number(label='hours_studied'),
        gr.Number(label='attendance'),
        gr.Number(label='past_grade'),
        gr.Radio(choices=[("Yes", 1), ("No", 0)], label="Assignments Completed")
    ],
    outputs='text',
    title="Student Exam Result Predictor",
    description="Predict if a student will pass or fail based on their academic data."
)

interface.launch()