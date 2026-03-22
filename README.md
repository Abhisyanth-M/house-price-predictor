# Bangalore House Price Predictor

A Machine Learning web app that predicts fair house prices in Bangalore based on location, size, BHK and bathrooms — helping buyers make informed decisions and avoid broker overpricing.

## Live Demo
https://huggingface.co/spaces/Abhisyanth-M/house-price-predictor

## Problem Statement
Bangalore real estate is confusing and overpriced. Buyers have no way to verify if a broker's quoted price is fair or inflated. A 2BHK in Koramangala and a 2BHK in Yelahanka can differ by lakhs — but most buyers don't know this.

## Solution
An ML-powered price prediction app that takes location, square feet, BHK and bathrooms as input and predicts the fair market price instantly — giving buyers data to negotiate confidently with brokers.

## Features
- Select from 135 Bangalore locations
- Input square feet, BHK and bathrooms
- Instant fair price prediction in lakhs
- Price per square foot calculation
- Sidebar showing model accuracy and dataset info
- Trained on 6,950 real Bangalore house listings

## Tech Stack
- Python
- Scikit-learn
- Linear Regression
- Pandas
- NumPy
- Streamlit

## Dataset
- Source: Kaggle — Bengaluru House Price Data by Amit Abhajoy
- Size: 6,950 house listings
- Locations: 135 areas across Bangalore

## ML Model
- Algorithm: Linear Regression
- Accuracy: R² score of 0.81
- Features: location, total square feet, bathrooms, BHK
- Encoding: One-hot encoding for location

## Key Insights
- Koramangala average price — Rs 107 lakhs for 2BHK 1000 sqft
- Yelahanka average price — Rs 41 lakhs for 2BHK 1000 sqft
- Location is the strongest predictor of house price in Bangalore

## How to Run Locally
```bash
git clone https://github.com/Abhisyanth-M/house-price-predictor
cd house-price-predictor
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Limitations
- Dataset is from 2017-2019 — prices have increased significantly since then
- Does not account for amenities, floor number, age of building
- Predictions are approximate — actual prices depend on many additional factors

## GitHub
https://github.com/Abhisyanth-M/house-price-predictor
