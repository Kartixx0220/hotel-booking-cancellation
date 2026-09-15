import streamlit as st
import pandas as pd
import joblib


# =========================
# Load saved artifacts
# =========================

rf_model = joblib.load("models/random_forest_final.pkl")
threshold = joblib.load("models/random_forest_threshold.pkl")
onehot_encoder = joblib.load("models/onehot_encoder.pkl")

agent_freq = joblib.load("models/agent_frequency_map.pkl")
country_freq = joblib.load("models/country_frequency_map.pkl")

feature_columns = joblib.load("models/feature_columns.pkl")

cat_cols = [
    'hotel',
    'arrival_date_month',
    'meal',
    'market_segment',
    'distribution_channel',
    'is_repeated_guest',
    'reserved_room_type',
    'assigned_room_type',
    'deposit_type',
    'customer_type'
]

num_cols = [
    'lead_time',
    'arrival_date_day_of_month',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'previous_cancellations',
    'previous_bookings_not_canceled',
    'booking_changes',
    'days_in_waiting_list',
    'adr',
    'required_car_parking_spaces',
    'total_of_special_requests'
]

freq_cols = [
    'agent',
    'country'
]

def predict_cancellation(booking):

    # Convert user input into a one-row DataFrame
    df = pd.DataFrame([booking])

    # -------------------------
    # Frequency encoding
    # -------------------------
    df['agent'] = df['agent'].map(agent_freq).fillna(0)
    df['country'] = df['country'].map(country_freq).fillna(0)

    # -------------------------
    # One-hot encoding
    # -------------------------
    encoded = onehot_encoder.transform(df[cat_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=onehot_encoder.get_feature_names_out(cat_cols),
        index=df.index
    )

    # -------------------------
    # Combine features
    # -------------------------
    X_final = pd.concat(
        [
            df[num_cols + freq_cols],
            encoded_df
        ],
        axis=1
    )

    # Ensure exact training feature order
    X_final = X_final.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # -------------------------
    # Prediction
    # -------------------------
    probability = rf_model.predict_proba(X_final)[0, 1]

    prediction = int(probability >= threshold)

    return probability, prediction


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Hotel Booking Cancellation Predictor")
st.write(
    "Enter the booking details below to predict whether the booking "
    "is likely to be canceled."
)

st.divider()

# =========================
# Booking Information
# =========================

st.subheader("📋 Booking Information")

col1, col2, col3 = st.columns(3)

with col1:
    hotel = st.selectbox(
        "Hotel",
        ["Resort Hotel", "City Hotel"]
    )

with col2:
    arrival_date_month = st.selectbox(
        "Arrival Month",
        [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ]
    )

with col3:
    arrival_date_day_of_month = st.number_input(
        "Arrival Day",
        min_value=1,
        max_value=31,
        value=15
    )


col1, col2, col3 = st.columns(3)

with col1:
    lead_time = st.number_input(
        "Lead Time (days)",
        min_value=0,
        value=100
    )

with col2:
    market_segment = st.selectbox(
        "Market Segment",
        [
            "Online TA",
            "Offline TA/TO",
            "Groups",
            "Direct",
            "Corporate",
            "Complementary",
            "Aviation",
            "Undefined"
        ]
    )

with col3:
    distribution_channel = st.selectbox(
        "Distribution Channel",
        [
            "TA/TO",
            "Direct",
            "Corporate",
            "GDS",
            "Undefined"
        ]
    )

# =========================
# Stay Information
# =========================

st.subheader("🛏️ Stay Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    stays_in_weekend_nights = st.number_input(
        "Weekend Nights",
        min_value=0,
        value=1
    )

with col2:
    stays_in_week_nights = st.number_input(
        "Week Nights",
        min_value=0,
        value=3
    )

with col3:
    adults = st.number_input(
        "Adults",
        min_value=0,
        value=2
    )

with col4:
    children = st.number_input(
        "Children",
        min_value=0,
        value=0
    )


col1, col2, col3, col4 = st.columns(4)

with col1:
    babies = st.number_input(
        "Babies",
        min_value=0,
        value=0
    )

with col2:
    adr = st.number_input(
        "Average Daily Rate (ADR)",
        min_value=0.0,
        value=100.0
    )

with col3:
    required_car_parking_spaces = st.number_input(
        "Parking Spaces Required",
        min_value=0,
        value=0
    )

with col4:
    total_of_special_requests = st.number_input(
        "Special Requests",
        min_value=0,
        value=1
    )

# =========================
# Room & Meal Information
# =========================

st.subheader("🛎️ Room & Booking Details")

col1, col2, col3 = st.columns(3)

with col1:
    meal = st.selectbox(
        "Meal",
        ["BB", "HB", "FB", "SC", "Undefined"]
    )

with col2:
    reserved_room_type = st.selectbox(
        "Reserved Room Type",
        [
            "A", "B", "C", "D", "E",
            "F", "G", "H", "L", "P"
        ]
    )

with col3:
    assigned_room_type = st.selectbox(
        "Assigned Room Type",
        [
            "A", "B", "C", "D", "E",
            "F", "G", "H", "I", "K", "L", "P"
        ]
    )


col1, col2 = st.columns(2)

with col1:
    deposit_type = st.selectbox(
        "Deposit Type",
        [
            "No Deposit",
            "Non Refund",
            "Refundable"
        ]
    )

with col2:
    customer_type = st.selectbox(
        "Customer Type",
        [
            "Transient",
            "Transient-Party",
            "Contract",
            "Group"
        ]
    )

# =========================
# Guest History
# =========================

st.subheader("👤 Guest & Booking History")

col1, col2, col3, col4 = st.columns(4)

with col1:
    is_repeated_guest = st.selectbox(
        "Repeated Guest",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:
    previous_cancellations = st.number_input(
        "Previous Cancellations",
        min_value=0,
        value=0
    )

with col3:
    previous_bookings_not_canceled = st.number_input(
        "Previous Non-Canceled Bookings",
        min_value=0,
        value=0
    )

with col4:
    booking_changes = st.number_input(
        "Booking Changes",
        min_value=0,
        value=0
    )


col1, col2, col3 = st.columns(3)

with col1:
    days_in_waiting_list = st.number_input(
        "Days in Waiting List",
        min_value=0,
        value=0
    )

with col2:
    country = st.text_input(
        "Country Code",
        value="PRT"
    ).upper()

with col3:
    agent = st.number_input(
        "Agent ID",
        min_value=0,
        value=9
    )

# =========================
# Prediction
# =========================

st.divider()

if st.button(
    "🔮 Predict Cancellation",
    type="primary",
    use_container_width=True
):

    booking = {
        'hotel': hotel,
        'arrival_date_month': arrival_date_month,
        'meal': meal,
        'country': country,
        'market_segment': market_segment,
        'distribution_channel': distribution_channel,
        'is_repeated_guest': is_repeated_guest,
        'reserved_room_type': reserved_room_type,
        'assigned_room_type': assigned_room_type,
        'deposit_type': deposit_type,
        'customer_type': customer_type,

        'lead_time': lead_time,
        'arrival_date_day_of_month': arrival_date_day_of_month,
        'stays_in_weekend_nights': stays_in_weekend_nights,
        'stays_in_week_nights': stays_in_week_nights,
        'adults': adults,
        'children': children,
        'babies': babies,
        'previous_cancellations': previous_cancellations,
        'previous_bookings_not_canceled': previous_bookings_not_canceled,
        'booking_changes': booking_changes,
        'days_in_waiting_list': days_in_waiting_list,
        'adr': adr,
        'required_car_parking_spaces': required_car_parking_spaces,
        'total_of_special_requests': total_of_special_requests,

        'agent': agent
    }

    probability, prediction = predict_cancellation(booking)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Booking is likely to be CANCELLED")
    else:
        st.success("✅ Booking is likely to NOT be cancelled")

    st.metric(
        "Cancellation Probability",
        f"{probability:.1%}"
    )

    st.progress(float(probability))

    st.caption(
        f"Model threshold: {threshold:.1%}"
    )