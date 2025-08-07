import pandas as pd
import gradio as gr
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

data = load_iris()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target
X = df.drop(['target'], axis=1)
y = df['target']

print(df)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


selector = SelectKBest(score_func=f_classif, k=2)
x_train_new = selector.fit_transform(x_train, y_train)
x_test_new = selector.transform(x_test)

model = RandomForestClassifier()

scores = cross_val_score(model, x_train_new, y_train, cv=5)

params = {
    'n_estimators':[10, 50, 100],
    'max_depth':[2,4,6]
}
grid_search = GridSearchCV(model, param_grid=params, cv=3)
grid_search.fit(x_train_new, y_train)
best_model = grid_search.best_estimator_


# Prediction function
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    input_new = selector.transform(input_data)
    prediction = best_model.predict(input_new)[0]
    prob = best_model.predict_proba(input_new)[0]
    return f"{data.target_names[prediction]} (Confidence: {max(prob)*100:.2f}%)"



# Gradio Interface
interface = gr.Interface(
    fn=predict_species,
    inputs=[
        gr.Number(label="Sepal Length (cm)"),
        gr.Number(label="Sepal Width (cm)"),
        gr.Number(label="Petal Length (cm)"),
        gr.Number(label="Petal Width (cm)")
    ],
    outputs=gr.Text(label="Predicted Species"),
    title="Iris Flower Classifier",
    description="Enter flower features to predict its species using RandomForest."
)

interface.launch()