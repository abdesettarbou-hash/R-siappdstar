import streamlit as st
from qcm.generator import generate_qcms
import pandas as pd

st.set_page_config(page_title="Générateur QCM - Résidanat", layout="wide")

st.title("Générateur de QCM — Résidanat Algérie")
st.caption("Génère des QCM avec l'API OpenAI.")

with st.form("params"):
    subject = st.text_input("Sujet", "Cardiologie - Insuffisance cardiaque")
    level = st.selectbox("Niveau", ["Facile", "Moyen", "Difficile"], index=1)
    n = st.slider("Nombre de questions", 5, 50, 10)
    submitted = st.form_submit_button("Générer")

if submitted:
    st.write("⚠️ Placeholder : branchement IA non encore implémenté")
