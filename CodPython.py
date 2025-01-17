
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

                                # Incarcare date
df = pd.read_csv("tourism.csv")  # se presupune ca fisierul este in acelasi director

df.dropna(inplace=True) 

df_encoded = pd.get_dummies(df, columns=["Country", "Category"], drop_first=True)

X = df_encoded.drop("Revenue", axis=1)
y = df_encoded["Revenue"]

X_train, X_test, y_train, y_test = train_test_split( #  Impart setul in train (80%) si test (20%)
    X, y,
    test_size=0.2,
    random_state=42
)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42) # Random Forest
model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test) # RMSE, MAE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print("Performanta Random Forest:")
print(f"RMSE = {rmse:.2f}")
print(f"MAE  = {mae:.2f}")

chosen_country = "Country_India"  # Cream un ranking pentru o tara exemplu (ex: 'Country_India')
year_avg = int(df["Year"].mean())
visitors_avg = int(df["Visitors"].mean())

possible_cats = [c for c in df_encoded.columns if c.startswith("Category_")]

# Generez un mini-set de date pentru cele 6 categorii
rows_for_inference = []
for cat_col in possible_cats:
    row_dict = {col: 0.0 for col in X.columns}  # totul 0 la inceput
    
    if chosen_country in row_dict:
        row_dict[chosen_country] = 1.0
    
    if cat_col in row_dict:
        row_dict[cat_col] = 1.0
    
    if "Year" in row_dict:   # Setez valori medii la 'Year' si 'Visitors'
        row_dict["Year"] = year_avg
    if "Visitors" in row_dict:
        row_dict["Visitors"] = visitors_avg
    
    rows_for_inference.append(row_dict)

inf_df = pd.DataFrame(rows_for_inference) # Convertesc lista de dict in DataFrame cu aceleasi coloane ca X
inf_df = inf_df[X.columns]

inf_preds = model_rf.predict(inf_df) 

ranking = sorted(zip(possible_cats, inf_preds), key=lambda x: x[1], reverse=True) # Sortez descrescator dupa venitul prezis

print(f"\nRanking categoriilor pentru {chosen_country}:")
for cat, val in ranking:
    cat_name = cat.replace("Category_", "")
    print(f"{cat_name}: {val:.2f}")
