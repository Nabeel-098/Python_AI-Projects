import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import gradio as gr

# Data
data = {
    "brand": ["Maruti", "Hyundai", "Ford", "BMW", "Audi", "Maruti", "Hyundai", "Ford", "BMW", "Audi"],
    "year": [2012, 2014, 2015, 2018, 2017, 2013, 2016, 2015, 2019, 2020],
    "kms_driven": [50000, 30000, 40000, 20000, 25000, 45000, 35000, 38000, 15000, 12000],
    "fuel_type": ["Petrol", "Diesel", "Diesel", "Petrol", "Diesel", "Petrol", "Diesel", "Diesel", "Petrol", "Petrol"],
    "transmission": ["Manual", "Manual", "Manual", "Automatic", "Automatic", "Manual", "Manual", "Manual", "Automatic", "Automatic"],
    "price": [2.5, 3.0, 3.5, 20.0, 22.0, 2.8, 3.2, 3.4, 21.0, 23.0]
}

df = pd.DataFrame(data)

# Label encoding
brand_map = {'Maruti': 0, 'Hyundai': 1, 'Ford': 2, 'BMW': 3, 'Audi': 4}
fuel_map = {'Petrol': 1, 'Diesel': 0}
trans_map = {'Manual': 0, 'Automatic': 1}

df['brand'] = df['brand'].map(brand_map)
df['fuel_type'] = df['fuel_type'].map(fuel_map)
df['transmission'] = df['transmission'].map(trans_map)

# Feature engineering
df['age'] = pd.Timestamp.now().year - df['year']
df.drop('year', axis=1, inplace=True)

# Split
x = df.drop('price', axis=1)
y = df['price']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

selector = SelectKBest(score_func=f_regression, k=4)
x_train_new = selector.fit_transform(x_train, y_train)
x_test_new = selector.transform(x_test)

model = RandomForestRegressor()
params = {'n_estimators': [10, 50], 'max_depth': [2, 4]}
grid_search = GridSearchCV(model, param_grid=params, cv=3)
grid_search.fit(x_train_new, y_train)
best_model = grid_search.best_estimator_

selected_features = x.columns[selector.get_support()]

def checkprice(brand_name, year, kms_driven, fuel_type, transmission):
    # Prepare and preprocess
    age = pd.Timestamp.now().year - int(year)
    input_df = pd.DataFrame([[
        brand_map[brand_name],
        kms_driven,
        fuel_map[fuel_type],
        trans_map[transmission],
        age
    ]], columns=["brand", "kms_driven", "fuel_type", "transmission", "age"])

    input_scaled = scaler.transform(input_df)
    input_selected = selector.transform(input_scaled)
    prediction = best_model.predict(input_selected)[0]
    return f"Estimated Price: ₹{prediction:.2f} lakhs"

# Gradio UI
interface = gr.Interface(
    fn=checkprice,
    inputs=[
        gr.Dropdown(choices=list(brand_map.keys()), label="Select Brand"),
        gr.Number(label='How Old is your car'),
        gr.Number(label='Kms Driven'),
        gr.Dropdown(choices=list(fuel_map.keys()), label="Fuel Type"),
        gr.Dropdown(choices=list(trans_map.keys()), label="Transmission")
    ],
    outputs='text',
    title="Used Car Price Predictor",
    description="Predict the price of a used car based on its features."
)

interface.launch()
