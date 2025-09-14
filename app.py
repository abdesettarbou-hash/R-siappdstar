import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from openai import OpenAI

st.set_page_config(page_title="Résidanat QCM", layout="wide")

# ===== API OPENAI =====
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.warning("⚠️ Clé API OpenAI manquante. Ajoutez-la dans Streamlit Cloud → Settings → Secrets.")

# ===== Interface =====
st.title("📚 Générateur & Tracker de QCM - Résidanat")

menu = st.sidebar.radio("Navigation", ["📖 Générer QCM", "📊 Suivi résultats"])

# ===== Génération de QCM =====
if menu == "📖 Générer QCM":
    module = st.text_input("Module :", "Biochimie")
    theme = st.text_input("Thème :", "Métabolisme du glucose")
    n_qcm = st.slider("Nombre de QCM :", 1, 5, 3)

    if st.button("Générer"):
        if not api_key:
            st.error("Clé API manquante. Impossible de générer.")
        else:
            with st.spinner("Génération en cours..."):
                prompt = f"""
                Génère {n_qcm} QCM pour un concours de résidanat en {module}, thème : {theme}.
                Contraintes :
                - Chaque QCM contient 1 question + 5 propositions de réponse (A à E).
                - Une seule bonne réponse.
                - Donne la correction et une explication claire.
                - Format strict : Q1, Q2, etc.
                """

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Tu es un expert en médecine et pédagogie."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4
                )
                st.markdown(response.choices[0].message.content)

# ===== Suivi des résultats =====
if menu == "📊 Suivi résultats":
    st.subheader("📥 Importer fichier Excel avec résultats")
    file = st.file_uploader("Choisir fichier (.xlsx)", type=["xlsx"])

    if file:
        df = pd.read_excel(file)
        st.dataframe(df)

        # Graph simple : progression globale
        if "Date" in df.columns and "%" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
            fig, ax = plt.subplots()
            ax.plot(df["Date"], df["%"], marker="o")
            ax.set_title("Progression globale (%)")
            st.pyplot(fig)

        # Forecast (projection linéaire)
        if len(df) > 2:
            df["days"] = (df["Date"] - df["Date"].min()).dt.days
            X = df[["days"]]
            y = df["%"]

            model = LinearRegression()
            model.fit(X, y)

            future_days = np.arange(df["days"].max()+1, df["days"].max()+11).reshape(-1, 1)
            y_pred = model.predict(future_days)
            future_dates = [df["Date"].max() + pd.Timedelta(days=i) for i in range(1, 11)]

            fig2, ax2 = plt.subplots()
            ax2.plot(df["Date"], y, "bo-", label="Réel")
            ax2.plot(future_dates, y_pred, "r--", label="Projection")
            ax2.legend()
            ax2.set_title("Projection des performances (10 jours)")
            st.pyplot(fig2)
