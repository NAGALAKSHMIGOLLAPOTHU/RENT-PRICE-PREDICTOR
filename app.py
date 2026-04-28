import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open('model/model.pkl', 'rb'))
columns = pickle.load(open('model/columns.pkl', 'rb'))

st.set_page_config(page_title="Rent Predictor", layout="centered")

st.title("🏠 Rent Price Predictor (India)")
st.write("💡 Enter house details to estimate rent")

# Inputs
bhk = st.slider("BHK", 1, 5)
size = st.number_input("Size (sqft)", 200, 5000)
floor = st.number_input("Floor", 0, 50)
bathroom = st.slider("Bathrooms", 1, 5)

area_type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"])
furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
tenant = st.selectbox("Tenant Preferred", ["Bachelors", "Family"])

# Input dictionary
input_data = {
    'BHK': bhk,
    'Size': size,
    'Floor': floor,
    'Bathroom': bathroom,
    'Area Type': area_type,
    'City': city,
    'Furnishing Status': furnishing,
    'Tenant Preferred': tenant
}

# Convert to dataframe
df = pd.DataFrame([input_data])
df = pd.get_dummies(df)

# Match training columns
df = df.reindex(columns=columns, fill_value=0)

# Prediction
if st.button("Predict Rent"):
    prediction = model.predict(df)[0]

    st.success(f"💰 Estimated Rent: ₹{int(prediction)}")

    # Price comparison
    actual = st.number_input("Enter Actual Rent to Compare", min_value=0)

    if actual > 0:
        if actual > prediction * 1.1:
            st.error("Overpriced ❌")
        elif actual < prediction * 0.9:
            st.success("Good Deal ✅")
        else:
            st.info("Fair Price 👍")