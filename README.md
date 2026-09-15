# 🏨 Hotel Booking Cancellation Prediction

A machine learning project that predicts whether a hotel booking is likely to be canceled.

I built this project to explore a complete machine learning workflow — starting from data cleaning and exploratory analysis, moving through feature engineering and model comparison, and finally deploying the selected model as an interactive Streamlit application.

## What does this project do?

The application takes information about a hotel booking, such as:

- Hotel type
- Lead time
- Arrival date
- Length of stay
- Number of guests
- Room type
- Deposit type
- Market segment
- Customer type
- Previous booking history
- Special requests
- Average Daily Rate (ADR)

and predicts the probability that the booking will be canceled.

The final application is designed to give a simple result:

**Cancellation Probability → Predicted Cancellation / No Cancellation**

---

## 📊 Dataset

This project uses the **Hotel Booking Demand** dataset.

The dataset contains **119,390 hotel booking records** from a Resort Hotel and a City Hotel, with bookings covering July 2015 to August 2017.

The dataset was originally published as part of the research paper:

> *Hotel Booking Demand Datasets*

by Nuno Antonio, Ana Almeida, and Luis Nunes.

The dataset is also widely available through Kaggle and other data-science repositories.

The target variable used in this project is:

```text
is_canceled
```

where:

- `0` → Booking was not canceled
- `1` → Booking was canceled

The original dataset contains 32 columns. Some of them were removed during preprocessing because they were not used in the final prediction pipeline.

---

## 🔧 Data Preparation

Before training the models, I cleaned and prepared the dataset for machine learning.

Some of the main preprocessing steps included:

- Handling missing values
- Replacing `Undefined` meal values
- Removing columns that were not suitable for the final model
- Removing invalid ADR values
- Encoding categorical variables
- Frequency encoding `agent` and `country`
- One-hot encoding the remaining categorical features

The final model uses **76 features** after preprocessing.

An important part of the deployment is that the same preprocessing used during training is preserved through saved artifacts. This means the Streamlit application can take raw booking information and transform it in the same way as the training data.

---

## 🤖 Models Compared

I experimented with several classification models:

- Logistic Regression
- Random Forest
- HistGradientBoosting
- XGBoost

After comparing their performance, **Random Forest** was selected as the final model for deployment.

---

## 🌲 Final Model Performance

The final Random Forest model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 89.39% |
| Precision | 88.57% |
| Recall | 81.94% |
| F1 Score | 85.13% |
| ROC-AUC | 95.92% |

The classification threshold used by the application is **0.5**.

This means that if the predicted cancellation probability is at least 50%, the application classifies the booking as likely to be canceled.

---

## 🔍 Model Interpretation

I also used SHAP-based analysis to get a better understanding of which features were influencing the Random Forest predictions.

Some of the features that appeared among the most influential were:

- Country
- Total number of special requests
- Lead time
- Deposit type
- Agent
- Booking changes
- ADR
- Market segment
- Previous cancellations

These results describe patterns learned by the model. They should not be interpreted as proof that a particular feature directly causes a booking to be canceled.

---

## 🌐 Streamlit Application

The trained model is wrapped in a Streamlit application.

The user can enter booking information through the web interface, and the application performs the following steps:

```text
User Input
    ↓
Frequency Encoding
    ↓
One-Hot Encoding
    ↓
Feature Alignment
    ↓
Random Forest
    ↓
Cancellation Probability
    ↓
Final Prediction
```

The application does **not** retrain the model. It loads the trained model and preprocessing artifacts that were created during the training process.

---

## 📁 Project Structure

```text
hotel-booking-cancellation/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── random_forest_final.pkl
│   ├── random_forest_threshold.pkl
│   ├── onehot_encoder.pkl
│   ├── agent_frequency_map.pkl
│   ├── country_frequency_map.pkl
│   └── feature_columns.pkl
│
└── notebooks/
    └── hotel_booking_cancellation.ipynb
```

### `app.py`
The Streamlit application and prediction pipeline.

### `models/`
Contains the trained Random Forest and the preprocessing artifacts required for prediction.

### `notebooks/`
Contains the notebook used for data analysis, preprocessing, model training, evaluation, and experimentation.

---

## ⚙️ Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

---

## ▶️ Running the Project Locally

Clone the repository:

```bash
git clone <your-repository-url>
cd hotel-booking-cancellation
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will then be available locally in your browser.

---

## 🚀 Live Demo

The application will be deployed using Streamlit Community Cloud.

**Live App:**  
`<add your Streamlit URL here>`

---

## 👨‍💻 About the Project

This project was built as a hands-on machine learning project to understand the complete process of taking a dataset from preprocessing and model experimentation to a working deployed application.