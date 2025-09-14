import streamlit as st
import pandas as pd

# Import sécurisé pour éviter crash si qcm n'est pas trouvé
try:
    from qcm.generator import generate_qcms
except ModuleNotFoundError:
    def generate_qcms(*args, **kwargs):
        return []

st.set_page_config(page_title="Générateur QCM - Résidanat", layout="wide")

st.title("Générateur de QCM — Résidanat Algérie")
st.caption("Génère des QCM avec l'API OpenAI.")

with st.form("params"):
    subject = st.text_input("Sujet", "Cardiologie - Insuffisance cardiaque")
    level = st.selectbox("Niveau", ["Facile", "Moyen", "Difficile"], index=1)
    n = st.slider("Nombre de questions", 5, 50, 10)
    submitted = st.form_submit_button("Générer")

if submitted:
    qcms = generate_qcms(subject=subject, n=n, level=level)
    if not qcms:
        st.warning("⚠️ Placeholder : IA non encore branchée")
    else:
        st.success(f"{len(qcms)} QCM générés.")
        st.dataframe(pd.DataFrame(qcms))
