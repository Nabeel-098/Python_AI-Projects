import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score

df = pd.read_csv('creditcard.csv')
X = df.drop(['Class'], axis=1)
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

selector = SelectKBest(score_func=f_classif, k=10)
x_train_new = selector.fit_transform(x_train, y_train)
x_test_new = selector.transform(x_test)

weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
model = xgb.XGBClassifier(scale_pos_weight=weight)

scores = cross_val_score(model, x_train_new, y_train, cv=5)

params = {
    'n_estimators': [50, 100],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.05]
}
grid_search = GridSearchCV(model, param_grid=params, cv=3)
grid_search.fit(x_train_new, y_train)
best_model = grid_search.best_estimator_

print("AUC Score:", roc_auc_score(y_test, best_model.predict_proba(x_test_new)[:, 1]))
joblib.dump(best_model, 'fraud_model_xgb.pkl')