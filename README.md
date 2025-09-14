# QCM Résidanat — Fixed Enhanced FR

Cette version corrige les problèmes signalés:
- Génération de QCM (AI si clé présente) ou fallback
- Selectbox pour modules/chapitres (liste étendue ; possibilité de module personnalisé)
- QCM interactifs: sélection par clic, possibilité d'afficher ou masquer corrections
- Examen blanc: chrono, sélection par clic, bouton Soumettre pour correction globale
- Bank local (qcm_bank.json) et dashboard

## Installation locale
1. `pip install -r requirements.txt`
2. `streamlit run app.py`

## Déploiement
- Poussez ce repo sur GitHub et déployez avec Streamlit Cloud.
- Pour activer l'AI: ajout du secret `OPENAI_API_KEY` dans Streamlit Cloud.
