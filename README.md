# Résidanat QCM Generator - Final

Application Streamlit pour générer des QCM (concours de résidanat Algérie) en utilisant l'API OpenAI.

## Installation locale
```bash
git clone https://github.com/<ton-user>/residanat-qcm.git
cd residanat-qcm
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement (Streamlit Cloud)
1. Push sur GitHub.
2. Créer une app sur Streamlit Cloud et lier le repo.
3. Dans Streamlit Cloud → Settings → Secrets: ajouter `OPENAI_API_KEY = "ta_clef"`.

## Notes
- Toujours vérifier manuellement les QCM générés avant usage.
- L'application supporte sélection Module → Chapitre, génération via OpenAI, export CSV.
