import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

# 📌 Load Excel File
excel_path = "JEE dataset.xlsx"  # Ensure this file is in the same folder
if not os.path.exists(excel_path):
    raise FileNotFoundError("Place your JEE dataset Excel file in the project directory.")

xls = pd.ExcelFile(excel_path)
print("Available Sheets:", xls.sheet_names)

def clean_closing_rank(rank):
    try:
        return float(rank)
    except:
        return None

def load_roundwise_data(base_name, year, round_nums):
    df_list = []
    for r in round_nums:
        sheet_name = f"{base_name} round {r}"
        if sheet_name not in xls.sheet_names:
            print(f"⚠️ Sheet '{sheet_name}' not found. Skipping...")
            continue
        df = xls.parse(sheet_name)
        df = df.rename(columns=lambda x: x.strip())
        df['Closing Rank'] = df['Closing Rank'].apply(clean_closing_rank)
        df = df[df['Closing Rank'].notna()]
        df['Year'] = year
        df['Round'] = str(r)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# Load 2023 & 2024 Data
df_2023 = load_roundwise_data("Jossa 23", 2023, [1,2,3,4,5,6])
df_2024 = load_roundwise_data("Jossa 24", 2024, [1,2,3,4,5])

# Combine Data
df_all = pd.concat([df_2023, df_2024], ignore_index=True)
print("✅ Combined Data Shape:", df_all.shape)

# Label Encoding
categorical_cols = ['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Gender', 'Round']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col].astype(str))
    encoders[col] = le

# Train/Test Split
features = ['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Gender', 'Round', 'Year']
X = df_all[features]
y = df_all['Closing Rank']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("📉 Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))

# Sample Prediction
sample_input = {
    'Institute': 'Indian Institute of Technology Kharagpur',
    'Academic Program Name': 'Computer Science and Engineering (4 Years, Bachelor of Technology)',
    'Quota': 'AI',
    'Seat Type': 'OPEN',
    'Gender': 'Gender-Neutral',
    'Round': '1',
    'Year': 2025
}
input_df = pd.DataFrame([sample_input])
for col in categorical_cols:
    input_df[col] = encoders[col].transform(input_df[col].astype(str))
predicted_cutoff = model.predict(input_df[features])[0]
print(f"🔮 Predicted Closing Rank for 2025 (Round {sample_input['Round']}): {int(predicted_cutoff)}")

# Trend Visualization for IIT Bombay CSE
iit = 'Indian Institute of Technology Bombay'
branch = 'Computer Science and Engineering (4 Years, Bachelor of Technology)'

filtered = df_all[
    (df_all['Institute'] == encoders['Institute'].transform([iit])[0]) &
    (df_all['Academic Program Name'] == encoders['Academic Program Name'].transform([branch])[0])
]

filtered['Round'] = filtered['Round'].astype(int)
plt.figure(figsize=(10,6))
for year in sorted(filtered['Year'].unique()):
    roundwise = filtered[filtered['Year'] == year].groupby('Round')['Closing Rank'].mean()
    plt.plot(roundwise.index, roundwise.values, marker='o', label=f"{year}")

plt.title(f"🧮 Closing Rank Trend: {branch} at {iit}")
plt.xlabel("Round")
plt.ylabel("Average Closing Rank")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
