import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="JEE College Predictor", layout="wide")

# Load Excel
excel_path = "JEE dataset.xlsx"
if not os.path.exists(excel_path):
    st.error("❌ 'JEE dataset.xlsx' not found in the current folder.")
    st.stop()

xls = pd.ExcelFile(excel_path)

@st.cache_data
def load_roundwise_data(base_name, year, round_nums):
    df_list = []
    for r in round_nums:
        sheet_name = f"{base_name} round {r}"
        if sheet_name not in xls.sheet_names:
            continue
        df = xls.parse(sheet_name)
        df = df.rename(columns=lambda x: x.strip())
        df['Closing Rank'] = pd.to_numeric(df['Closing Rank'], errors='coerce')
        df = df[df['Closing Rank'].notna()]
        df['Year'] = year
        df['Round'] = str(r)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

df_2023 = load_roundwise_data("Jossa 23", 2023, [1,2,3,4,5,6])
df_2024 = load_roundwise_data("Jossa 24", 2024, [1,2,3,4,5])
df_all = pd.concat([df_2023, df_2024], ignore_index=True)

# Label Encoding
categorical_cols = ['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Gender', 'Round']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col].astype(str))
    encoders[col] = le

# Train Model
features = ['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Gender', 'Round', 'Year']
X = df_all[features]
y = df_all['Closing Rank']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ----------------------------
# Streamlit UI
st.title("🎓 JEE (Main/Advanced) College Predictor")
st.markdown("Get an estimated **closing rank** for your target college/program based on past data (2023–2024).")

col1, col2 = st.columns(2)

with col1:
    selected_college = st.selectbox("🏫 Institute", encoders['Institute'].classes_)
    selected_branch = st.selectbox("📚 Branch", encoders['Academic Program Name'].classes_)
    selected_quota = st.selectbox("🪪 Quota", encoders['Quota'].classes_)

with col2:
    selected_seat = st.selectbox("🪑 Seat Type", encoders['Seat Type'].classes_)
    selected_gender = st.selectbox("🚻 Gender", encoders['Gender'].classes_)
    selected_round = st.selectbox("🔄 Round", sorted(encoders['Round'].classes_))
    selected_year = st.slider("📅 Target Year", 2025, 2026, 2025)

input_dict = {
    'Institute': selected_college,
    'Academic Program Name': selected_branch,
    'Quota': selected_quota,
    'Seat Type': selected_seat,
    'Gender': selected_gender,
    'Round': selected_round,
    'Year': selected_year
}

input_df = pd.DataFrame([input_dict])
for col in categorical_cols:
    input_df[col] = encoders[col].transform(input_df[col].astype(str))

# Predict
predicted_rank = model.predict(input_df[features])[0]
st.success(f"🔮 **Predicted Closing Rank**: {int(predicted_rank)}")

# Optional Trend Plot
st.subheader("📈 Trend of Closing Ranks Over Rounds (Past Years)")
show_plot = st.checkbox("Show trend plot", value=True)
if show_plot:
    filtered = df_all[
        (df_all['Institute'] == encoders['Institute'].transform([selected_college])[0]) &
        (df_all['Academic Program Name'] == encoders['Academic Program Name'].transform([selected_branch])[0])
    ]
    filtered['Round'] = filtered['Round'].astype(int)
    fig, ax = plt.subplots(figsize=(10,5))
    for year in sorted(filtered['Year'].unique()):
        group = filtered[filtered['Year'] == year].groupby('Round')['Closing Rank'].mean()
        ax.plot(group.index, group.values, marker='o', label=f"{year}")
    ax.set_title(f"{selected_branch} at {selected_college}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Avg. Closing Rank")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
