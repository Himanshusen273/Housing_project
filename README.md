# 🏡 California Housing Price Predictor
A modern Machine Learning web application built with **Gradio**, **XGBoost**, **Pandas**, and **SQLite** to predict median house values across California using block-level demographic and geographic data.

This project includes:
- A fully interactive UI
- A trained ML model (XGBoost)
- A preprocessing pipeline
- Real-time predictions
- Admin panel with secure password access
- SQLite database for storing user prediction history

---

## 🚀 Features

### ⭐ Machine Learning
- XGBoost Regressor trained on the California Housing Dataset
- Custom preprocessing pipeline using:
  - OneHot Encoding
  - Standard Scaling
  - Imputation (Median strategy)

### ⭐ Modern UI (Gradio)
- Clean, attractive gradient UI
- Sliders, number inputs, dropdowns
- Auto-formatted prediction card with metrics
- Example input presets

### ⭐ Database Integration (SQLite)
- Automatically stores every prediction in `housing_data.db`
- Includes:
  - Coordinates
  - Demographics
  - Property stats
  - Predicted value
  - Timestamp

### ⭐ Admin Panel
- Protected with password (change inside code)
- Displays full prediction logs from database
- Prevents unauthorized access

---

## 📂 Project Structure

project/
│── main.py # Main application
│── model.pkl # Trained ML model (ignored on GitHub)
│── pipeline.pkl # Preprocessing pipeline
│── housing_data.db # SQLite database (ignored on GitHub)
│── housing.csv # Dataset (optional)
│── requirements.txt # Dependencies
│── .gitignore # Ignore sensitive/heavy files
└── README.md # Documentation




---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repo
```bash
git clone <your-repo-url>
cd <your-repo-folder>


pip install -r requirements.txt

## Run the file 
python main.py



