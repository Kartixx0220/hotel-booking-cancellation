# 🏨 Hotel Booking Cancellation Predictor

A machine learning project that predicts whether a hotel booking is likely to be cancelled based on booking, guest, stay, and reservation details.

The project covers the complete machine learning workflow — from data cleaning and exploratory analysis to feature engineering, model comparison, interpretation, and deployment as an interactive Streamlit application.

## 🚀 Live Demo

Try the deployed application here:

👉 **[Hotel Booking Cancellation Predictor](https://hotel-booking-cancellation-sftslafwitgv7whpgem4lk.streamlit.app/)**

The application allows users to enter booking information and get:

- Cancellation probability
- Predicted cancellation status
- A simple visual representation of the prediction probability

---

## 📌 Project Overview

Hotel cancellations can affect room availability, revenue, and operational planning. Being able to identify bookings that are more likely to be cancelled can help hotels make better decisions around inventory and reservation management.

In this project, I built a binary classification model where:

- **Target = 1:** Booking is cancelled
- **Target = 0:** Booking is not cancelled

Several machine learning models were evaluated before selecting the final model based on overall predictive performance.

The final model is a **Random Forest Classifier**.

---

## 📊 Dataset

The project uses the **Hotel Booking Demand** dataset, containing hotel reservation information for both a Resort Hotel and a City Hotel.

The original dataset is associated with the research work:

> Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel Booking Demand Datasets.

The dataset contains information about:

- Hotel type
- Booking lead time
- Arrival date
- Length of stay
- Number of guests
- Meal plan
- Country
- Market segment
- Distribution channel
- Previous cancellations and bookings
- Room types
- Deposit type
- Customer type
- Average Daily Rate (ADR)
- Special requests
- Parking requirements
- Booking changes

The original dataset contains approximately **119,390 records and 32 columns**.

---

## 🧹 Data Preparation

Several data-cleaning and preprocessing steps were performed before model training.

### Missing Values

- Missing values in `children` were treated as `0`.
- `company` was removed because of its very high number of missing values.
- Other categorical and numerical variables were examined for missing values.

### Data Cleaning

The following cleaning decisions were made:

- `Undefined` values in `meal` were treated as `SC`.
- Rows with negative `adr` values were removed.
- The extreme `adr = 5400` observation was removed.
- `reservation_status` was removed.
- `reservation_status_date` was removed.

These last two variables were excluded because they describe the reservation outcome/status and could introduce information that would not realistically be available when making an early cancellation prediction.

---

## ⚙️ Feature Engineering

The final model uses numerical, categorical, and frequency-encoded features.

### Numerical Features

```text
lead_time
arrival_date_day_of_month
stays_in_weekend_nights
stays_in_week_nights
adults
children
babies
previous_cancellations
previous_bookings_not_canceled
booking_changes
days_in_waiting_list
adr
required_car_parking_spaces
total_of_special_requests
```

### Categorical Features

```text
hotel
arrival_date_month
meal
market_segment
distribution_channel
is_repeated_guest
reserved_room_type
assigned_room_type
deposit_type
customer_type
```

### Frequency Encoding

`agent` and `country` were handled using frequency encoding.

The frequency mappings were calculated **only from the training data** and then applied to the test data. This prevents information from the test set from leaking into the training process.

Unknown values during deployment are mapped to `0`.

### One-Hot Encoding

The remaining categorical variables were transformed using `OneHotEncoder`.

The final feature set contains **76 features**.

---

## 🤖 Models Evaluated

Multiple classification approaches were considered during the project, including:

- Logistic Regression
- Random Forest
- HistGradientBoosting
- XGBoost

The models were compared using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

After comparing the models, **Random Forest** was selected as the final model.

---

## 🌲 Final Model

The final model is an un-tuned Random Forest Classifier:

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

The model was selected based on its overall performance while maintaining a strong balance between precision, recall, F1-score, and ROC-AUC.

### Final Performance

| Metric | Score |
|---|---:|
| Accuracy | **89.39%** |
| Precision | **88.57%** |
| Recall | **81.94%** |
| F1-Score | **85.13%** |
| ROC-AUC | **95.92%** |

These results were obtained on the held-out test set.

---

## 🔍 Model Interpretation

Feature importance and SHAP-based analysis were used to understand which variables contributed most to the model's predictions.

Because the final Random Forest is relatively large, the SHAP analysis was performed using **approximate SHAP values** on a sample of the data.

Some of the most influential features included:

- Country
- Total number of special requests
- Lead time
- Deposit type
- Agent
- Booking changes
- ADR
- Market segment
- Previous cancellations

This analysis helped provide some insight into the factors associated with the model's cancellation predictions.

---

## 🎯 Prediction Threshold

The final prediction is based on the model's probability of cancellation.

The deployed application uses a threshold of:

```text
0.50
```

Therefore:

```text
Probability >= 0.50  →  Cancelled
Probability < 0.50   →  Not Cancelled
```

The application also displays the predicted cancellation probability to make the result easier to interpret.

---

## 🌐 Streamlit Application

The trained model has been integrated into an interactive Streamlit application.

Users can enter information such as:

### Booking Information

- Hotel
- Arrival month and day
- Lead time
- Market segment
- Distribution channel

### Stay Information

- Weekend nights
- Week nights
- Number of adults
- Children
- Babies
- ADR
- Parking spaces
- Special requests

### Room & Booking Details

- Meal
- Reserved room type
- Assigned room type
- Deposit type
- Customer type

### Guest & Booking History

- Repeated guest status
- Previous cancellations
- Previous non-cancelled bookings
- Booking changes
- Waiting list days
- Country
- Agent

The application then processes the input using the same preprocessing logic used during model development and returns the cancellation probability and predicted class.

---

## 📦 Model Deployment

The final Random Forest model is approximately **424 MB**.

GitHub has a **100 MB limit for individual files stored normally in a repository**, so the large model is not committed directly to the repository.

Instead, the final model:

```text
random_forest_final.pkl
```

is stored as a **GitHub Release asset**.

### Release

```text
Release: v1.0.0
Title: Initial Model Release
Asset: random_forest_final.pkl
```

The Streamlit application automatically downloads the model from the GitHub Release when the model is not already available locally.

This keeps the main repository lightweight while still allowing the deployed application to use the complete final Random Forest model.

The remaining smaller preprocessing artifacts are stored directly in the `models/` directory.

---

## 📁 Project Structure

```text
hotel-booking-cancellation/
│
├── app.py
├── README.md
├── requirements.txt
│
├── models/
│   ├── random_forest_threshold.pkl
│   ├── onehot_encoder.pkl
│   ├── agent_frequency_map.pkl
│   ├── country_frequency_map.pkl
│   └── feature_columns.pkl
│
└── notebook/
    └── hotel_booking_cancellation.ipynb
```

The final Random Forest model is stored separately as a GitHub Release asset:

```text
GitHub Release v1.0.0
└── random_forest_final.pkl
```

---

## 💻 Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Kartixx0220/hotel-booking-cancellation.git
```

### 2. Navigate into the project

```bash
cd hotel-booking-cancellation
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

On the first run, the application may take some time to download the large Random Forest model from the GitHub Release.

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost

### Model Interpretation

- SHAP

### Deployment

- Streamlit
- GitHub
- GitHub Releases

### Development Environment

- Jupyter Notebook
- VS Code

---

## 📚 What This Project Covers

This project was built as an end-to-end machine learning workflow rather than just a model-training exercise.

It covers:

- Exploratory Data Analysis
- Data Cleaning
- Missing Value Handling
- Feature Engineering
- Frequency Encoding
- One-Hot Encoding
- Train/Test Splitting
- Classification
- Model Comparison
- Model Evaluation
- Feature Importance
- Approximate SHAP Analysis
- Prediction Thresholds
- Model Serialization
- Streamlit Deployment
- GitHub-based Model Management

---

## 🔮 Possible Future Improvements

Some possible directions for further development include:

- More extensive hyperparameter optimization
- Probability calibration
- More detailed cost-based threshold selection
- Additional model interpretability features
- Monitoring model performance after deployment
- More advanced handling of high-cardinality categorical variables
- Further optimization of the model size for deployment

---

## 👤 Author

**Kartik Vats**

BTech — Computer Science

GitHub: [Kartixx0220](https://github.com/Kartixx0220)

---

## 📄 License

This project is intended for educational and portfolio purposes.
