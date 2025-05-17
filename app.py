from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

predicted_df = None  # Initialize as None

def load_predicted_data():
    global predicted_df
    CSV_FILE = 'predicted_dataset_with_2024.csv'
    if predicted_df is None:
        if os.path.exists(CSV_FILE):
            try:
                predicted_df = pd.read_csv(CSV_FILE)
                print(f"Successfully loaded: {CSV_FILE}")
            except FileNotFoundError:
                print(f"Error: {CSV_FILE} not found!")
                predicted_df = pd.DataFrame()
            except Exception as e:
                print(f"Error loading {CSV_FILE}: {e}")
        else:
            print(f"Warning: {CSV_FILE} does not exist in the current directory.")

@app.route('/predict', methods=['POST'])
def predict_colleges():
    load_predicted_data()  # Load data when the endpoint is hit

    if predicted_df is None or predicted_df.empty:
        return jsonify({'error': 'Predicted data not loaded'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    jee_rank = data.get('jee_rank')
    seat_type = data.get('seat_type')
    gender = data.get('gender', 'Gender-Neutral')
    domicile = data.get('domicile')
    program_preference = data.get('program_preference')
    institute_preference = data.get('institute_preference')
    quota = data.get('quota', 'AI')

    if jee_rank is None or seat_type is None or domicile is None or quota is None:
        return jsonify({'error': 'Missing required parameters (jee_rank, seat_type, domicile, quota)'}), 400

    filtered_df = predicted_df[predicted_df['Quota'] == quota]
    filtered_df = filtered_df[filtered_df['Seat Type'] == seat_type]
    filtered_df = filtered_df[filtered_df['Gender'] == gender]

    if program_preference:
        filtered_df = filtered_df[filtered_df['Academic Program Name'].str.contains(program_preference, case=False)]

    if institute_preference:
        filtered_df = filtered_df[filtered_df['Institute'].str.contains(institute_preference, case=False)]

    potential_colleges = []
    for index, row in filtered_df.iterrows():
        for round_num in range(1, 6): # Assuming 5 rounds
            round_col = f'Closing_Rank_2024_R{round_num}'
            if round_col in row and pd.notna(row[round_col]) and jee_rank <= row[round_col] * 1.10:
                potential_colleges.append({
                    'institute': row['Institute'],
                    'program': row['Academic Program Name'],
                    'quota': row['Quota'],
                    'seat_type': row['Seat Type'],
                    'gender': row['Gender'],
                    f'closing_rank_2024_round_{round_num}': row[round_col],
                    'average_closing_rank': row['Avg_Closing_Rank'],
                    'min_closing_rank': row['Min_Closing_Rank'],
                    'max_closing_rank': row['Max_Closing_Rank'],
                    'std_dev_closing_rank': row['Std_Dev_Closing_Rank']
                })
                break

    return jsonify(potential_colleges)

if __name__ == '__main__':
    app.run(debug=True)
