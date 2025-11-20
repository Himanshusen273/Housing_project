import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


# -----------------------------
# Build Preprocessing Pipeline
# -----------------------------
def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    return full_pipeline


# -----------------------------
# Initialize SQLite Database
# -----------------------------
def init_db():
    conn = sqlite3.connect("housing_data.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        longitude REAL,
        latitude REAL,
        housing_median_age REAL,
        total_rooms REAL,
        total_bedrooms REAL,
        population REAL,
        households REAL,
        median_income REAL,
        ocean_proximity TEXT,
        predicted_value REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


# -----------------------------
# Train Model Only First Time
# -----------------------------
def train_model():
    if not os.path.exists(MODEL_FILE):
        housing = pd.read_csv("housing.csv")

        housing['income_cat'] = pd.cut(
            housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5]
        )

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing['income_cat']):
            housing = housing.loc[train_index].drop("income_cat", axis=1)

        housing_labels = housing["median_house_value"].copy()
        housing_features = housing.drop("median_house_value", axis=1)

        num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
        cat_attribs = ["ocean_proximity"]

        pipeline = build_pipeline(num_attribs, cat_attribs)
        housing_prepared = pipeline.fit_transform(housing_features)

        model = XGBRegressor(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        model.fit(housing_prepared, housing_labels)

        joblib.dump(model, MODEL_FILE)
        joblib.dump(pipeline, PIPELINE_FILE)

        print("Model trained & saved!")
    else:
        print("Model already trained!")


# -----------------------------
# ADMIN PANEL FUNCTION
# -----------------------------
def view_admin_data(password):
    ADMIN_PASSWORD = "checkstatus8086"

    if password != ADMIN_PASSWORD:
        return "<h3 style='color:red;'>❌ Wrong Password</h3>"

    try:
        conn = sqlite3.connect("housing_data.db")
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
        conn.close()

        return df.to_html(classes='table table-bordered', index=False)

    except Exception as e:
        return f"<h3 style='color:red;'>Error: {str(e)}</h3>"


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_house_price(longitude, latitude, housing_median_age, total_rooms,
                        total_bedrooms, population, households, median_income,
                        ocean_proximity):

    try:
        model = joblib.load(MODEL_FILE)
        pipeline = joblib.load(PIPELINE_FILE)

        input_data = pd.DataFrame({
            'longitude': [float(longitude)],
            'latitude': [float(latitude)],
            'housing_median_age': [float(housing_median_age)],
            'total_rooms': [float(total_rooms)],
            'total_bedrooms': [float(total_bedrooms)],
            'population': [float(population)],
            'households': [float(households)],
            'median_income': [float(median_income)],
            'ocean_proximity': [str(ocean_proximity)]
        })

        transformed_input = pipeline.transform(input_data)
        prediction = float(model.predict(transformed_input)[0])

        # Save to SQL
        conn = sqlite3.connect("housing_data.db")
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO predictions
        (longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
        population, households, median_income, ocean_proximity, predicted_value)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            float(longitude), float(latitude), float(housing_median_age), float(total_rooms),
            float(total_bedrooms), float(population), float(households), float(median_income),
            str(ocean_proximity), prediction
        ))

        conn.commit()
        conn.close()

        return f"""
        <div style='padding:20px;text-align:center;border-radius:10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
        <h1 style='color:white;'>Predicted Price: ${prediction:,.0f}</h1>
        </div>
        """

    except Exception as e:
        return f"<div style='color:red;'>Error: {e}</div>"


# ============= APP STARTUP =============
init_db()
train_model()

custom_css = """
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
"""

with gr.Blocks(css=custom_css,
               theme=gr.themes.Soft(primary_hue="indigo")) as demo:

    gr.Markdown("# 🏡 California Housing Price Predictor")

    with gr.Row():
        with gr.Column():
            longitude = gr.Slider(-124.5, -114.0, -122.23, 0.01, label="Longitude")
            latitude = gr.Slider(32.5, 42.0, 37.88, 0.01, label="Latitude")
            ocean_proximity = gr.Dropdown(["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY"],
                                          label="Ocean Proximity")

        with gr.Column():
            housing_median_age = gr.Slider(1, 52, 41, 1, label="Housing Median Age")
            total_rooms = gr.Number(value=880, label="Total Rooms")
            total_bedrooms = gr.Number(value=129, label="Total Bedrooms")

        with gr.Column():
            population = gr.Number(value=322, label="Population")
            households = gr.Number(value=126, label="Households")
            median_income = gr.Slider(0.5, 15.0, 8.3252, 0.0001, label="Median Income")

    output = gr.HTML()
    predict_btn = gr.Button("🔮 Predict House Price", variant="primary")
    predict_btn.click(
        fn=predict_house_price,
        inputs=[
            longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income, ocean_proximity
        ],
        outputs=output
    )

    # ----------------- ADMIN PANEL -----------------
    gr.Markdown("---")
    gr.Markdown("## 🔐 Admin Panel")

    admin_password = gr.Textbox(label="Enter Admin Password", type="password")
    admin_output = gr.HTML()
    admin_btn = gr.Button("📂 View Database Records")

    admin_btn.click(
        fn=view_admin_data,
        inputs=admin_password,
        outputs=admin_output
    )

demo.launch(share=True)
