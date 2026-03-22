import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Bangalore House Price Predictor", page_icon="🏠")
st.title("🏠 Bangalore House Price Predictor")
st.markdown("Enter house details to get the predicted fair market price!")

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("Bengaluru_House_Data.csv")
    df = df.dropna()
    df = df[df["size"].str.contains("BHK", na=False)]
    df["bhk"] = df["size"].str.extract(r"(\d+)").astype(int)
    df = df[df["total_sqft"].str.replace(".", "", 1).str.isnumeric()]
    df["total_sqft"] = df["total_sqft"].astype(float)
    df = df[df["price_per_sqft"] if "price_per_sqft" in df.columns else df.index.isin(df.index)]
    df = df[["location", "total_sqft", "bath", "bhk", "price"]]
    df = df[df["total_sqft"] < 10000]
    df = df[df["price"] < 500]
    location_counts = df["location"].value_counts()
    df["location"] = df["location"].apply(lambda x: x if location_counts[x] > 10 else "Other")
    return df

df = load_and_clean_data()

@st.cache_resource
def train_model(df):
    X = pd.get_dummies(df[["location", "total_sqft", "bath", "bhk"]])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, X.columns.tolist(), r2

model, feature_columns, r2 = train_model(df)

st.sidebar.markdown("## Model Info")
st.sidebar.markdown("**Algorithm:** Linear Regression")
st.sidebar.markdown(f"**R² Score:** {r2:.2f}")
st.sidebar.markdown(f"**Total houses trained on:** {len(df):,}")
st.sidebar.markdown(f"**Locations available:** {df['location'].nunique()}")

st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Location", sorted(df["location"].unique()))
    total_sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, value=1000)

with col2:
    bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)

if st.button("Predict Price", type="primary"):
    input_data = pd.DataFrame([[location, total_sqft, bath, bhk]], 
                               columns=["location", "total_sqft", "bath", "bhk"])
    input_dummies = pd.get_dummies(input_data)
    input_aligned = input_dummies.reindex(columns=feature_columns, fill_value=0)
    predicted_price = model.predict(input_aligned)[0]
    
    if predicted_price < 0:
        st.error("Could not predict price for this combination. Try different values!")
    else:
        st.success(f"Predicted Price: ₹ {predicted_price:.2f} Lakhs")
        st.info(f"Price per sqft: ₹ {(predicted_price * 100000) / total_sqft:,.0f}")

