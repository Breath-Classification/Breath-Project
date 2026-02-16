import streamlit as st

from main import train_and_predict

st.title("ML Model Tester")

if st.button("Uruchom predykcję"):
    y_pred, y_true, X = train_and_predict()
    st.write("Predykcje:", y_pred)
    st.write("Prawdziwe:", y_true)