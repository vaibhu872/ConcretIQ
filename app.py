import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

API = "http://localhost:8000"

st.set_page_config(page_title="Cement Strength Predictor", layout="wide")
st.title("Cement Compressive Strength Predictor")
st.caption("Upgraded from LinearRegression → RandomForest/XGBoost with EDA + CV")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Mix design inputs")
    cement     = st.slider("Cement (kg/m³)",           100, 600, 300)
    slag       = st.slider("Blast furnace slag (kg/m³)", 0, 400, 0)
    fly_ash    = st.slider("Fly ash (kg/m³)",            0, 200, 0)
    water      = st.slider("Water (kg/m³)",             100, 250, 180)
    superp     = st.slider("Superplasticizer (kg/m³)",   0,  35, 5)
    coarse_agg = st.slider("Coarse aggregate (kg/m³)",  800, 1200, 1000)
    fine_agg   = st.slider("Fine aggregate (kg/m³)",    500, 1000, 800)
    age        = st.selectbox("Curing age (days)", [3, 7, 14, 28, 56, 90, 180, 365], index=3)

    if st.button("Predict strength", type="primary"):
        payload = dict(cement=cement, slag=slag, fly_ash=fly_ash,
                       water=water, superplasticizer=superp,
                       coarse_agg=coarse_agg, fine_agg=fine_agg, age=age)
        try:
            r = requests.post(f"{API}/predict", json=payload, timeout=5)
            if r.status_code == 200:
                result = r.json()
                st.session_state['result'] = result
            else:
                st.error(f"API error: {r.json().get('detail')}")
        except Exception as e:
            st.error(f"Could not reach API: {e}")

with col2:
    st.subheader("Prediction result")
    if 'result' in st.session_state:
        res = st.session_state['result']
        strength = res['predicted_strength_mpa']

        st.metric("Predicted strength", f"{strength} MPa")
        st.metric("Water/cement ratio", res['water_cement_ratio'])
        st.caption(res['confidence_note'])

        # Strength grade classification
        if strength < 20:   grade = "C15 — low strength"
        elif strength < 30: grade = "C25 — standard"
        elif strength < 40: grade = "C35 — high strength"
        elif strength < 55: grade = "C50 — very high"
        else:               grade = "C55+ — ultra high"
        st.success(f"Concrete grade: {grade}")

        # Simple bar gauge
        fig, ax = plt.subplots(figsize=(5, 1.5))
        ax.barh([''], [strength], color='steelblue')
        ax.barh([''], [80 - strength], left=[strength], color='#eee')
        ax.set_xlim(0, 80)
        ax.set_xlabel("MPa")
        ax.axvline(28, color='orange', linestyle='--', label='C25 threshold')
        ax.legend(fontsize=8)
        ax.set_title("Strength gauge (0–80 MPa)")
        plt.tight_layout()
        st.pyplot(fig)