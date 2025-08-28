Churn app: https://churnapp123.streamlit.app/

📊 Customer Churn Prediction App

This project is a Machine Learning Web App built with Streamlit to predict whether a customer is likely to churn (leave the service) or stay.
It uses a Random Forest Classifier trained on historical customer data.

🚀 Features

🔍 Predicts Customer Churn (Churn / No Churn)

📈 Displays Churn Probability

⚡ Scales user inputs with the same StandardScaler used during training

🌐 Simple and interactive Streamlit UI

📂 Project Structure
📦 churn-prediction-app
 ┣ 📜 app.py              
 ┣ 📜 train_model.py      
 ┣ 📜 rf_model.pkl       
 ┣ 📜 scaler.pkl        
 ┣ 📜 requirements.txt   
 ┗ 📜 README.md          

⚙️ Installation & Setup

1️⃣ Clone this repository:

git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app


2️⃣ Install dependencies:

pip install -r requirements.txt


3️⃣ Run the Streamlit app:

streamlit run app.py

🧪 How it Works

Data Preprocessing

StandardScaler is fitted on training data

Features are transformed using the same scaler

Model Training

Random Forest Classifier (rf_model.pkl) trained and saved

Scaler (scaler.pkl) also saved

Prediction in App

User enters details (e.g., tenure, charges, contract type)

Features are scaled with scaler.pkl

Model predicts churn and probability

🎯 Example Output

Prediction: No Churn

Churn Probability: 0.28

📌 Requirements

Python 3.8+

Streamlit

Scikit-learn

Numpy & Pandas

Joblib

🌟 Future Enhancements

📊 Add data visualization for churn trends

⚖️ Handle imbalanced dataset with SMOTE/threshold tuning

🏆 Add multiple ML models for comparison
