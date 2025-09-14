import streamlit as st
import pandas as pd
from qcm import generator as qgen

st.set_page_config(page_title="Générateur QCM - Résidanat", layout="wide")

st.title("Générateur de QCM — Résidanat Algérie")
st.markdown('Génère des QCM (questions à choix multiple) à l'aide d'une IA.')

# Modules / Chapitres (échantillon; étoffer selon besoin)
MODULES = {
    "Cardiologie": ["Insuffisance cardiaque", "Infarctus du myocarde", "HTA"],
    "Pneumologie": ["Asthme", "BPCO", "Tuberculose"],
    "Gastro-entérologie": ["Cirrhose", "Ulcère gastro-duodénal", "Hépatite virale"],
    "Neurologie": ["AVC", "Épilepsie", "Sclérose en plaques"]
}

with st.form("params"):
    module = st.selectbox("Module", list(MODULES.keys()))
    chapitre = st.selectbox("Chapitre", MODULES[module])
    niveau = st.selectbox("Niveau", ["Facile", "Moyen", "Difficile"], index=1)
    n = st.slider("Nombre de questions", 5, 50, 10)
    include_explanations = st.checkbox("Inclure explications courtes (1-2 lignes)", value=True)
    submitted = st.form_submit_button("Générer les QCM")

if submitted:
    sujet = f"{module} - {chapitre}"
    try:
        qcms = qgen.generate_qcms(subject=sujet, n=n, level=niveau, include_explanations=include_explanations)
    except Exception as e:
        st.error(f"Erreur lors de la génération: {e}")
        qcms = []

    if not qcms:
        st.warning("⚠️ Pas de questions générées (clé OPENAI manquante ou erreur). Vérifie les logs / secrets.")
    else:
        df = pd.DataFrame(qcms)
        # Normalize choices to columns for clearer view
        choices_expanded = pd.DataFrame(df['choices'].tolist(), columns=['A','B','C','D'])
        display_df = pd.concat([df.drop(columns=['choices']), choices_expanded], axis=1)
        st.success(f"{len(df)} questions générées.")
        st.dataframe(display_df)

        csv = df.to_csv(index=False)
        st.download_button("Télécharger CSV", csv, file_name="qcms.csv", mime="text/csv")

        st.subheader("Quiz interactif")
        score = 0
        for i, row in df.iterrows():
            st.markdown(f"**Q{i+1}.** {row['question']}")
            choice = st.radio(f"Choix (Q{i+1})", row['choices'], key=f"q{i}")
            if st.button("Valider", key=f"val{i}"):
                correct = row['answer']
                if choice == correct:
                    st.success("Correct ✅")
                    score += 1
                else:
                    st.error(f"Incorrect — bonne réponse: **{correct}**")
                if row.get('explanation'):
                    st.info(row.get('explanation'))

        st.info(f"Score total (interactif): {score} / {len(df)}")
