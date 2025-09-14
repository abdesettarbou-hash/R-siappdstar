# QCM Résidanat — Enhanced (FR)

Application Streamlit en français pour :
- Générer des QCM (OpenAI si configuré) ou via générateur de secours
- Pratiquer avec QCM interactifs (sélection par module/chapter, réponses par clic)
- Faire des examens blancs (chrono, correction à la fin)
- Bank local de QCM (JSON) et Dashboard basique

## Installation locale
1. `pip install -r requirements.txt`
2. `streamlit run app.py`

## Déploiement sur Streamlit Cloud
1. Poussez le repo sur GitHub (app.py, requirements.txt, README.md)
2. Dans Streamlit Cloud, créez une nouvelle app et reliez le repo
3. Ajoutez la clé OpenAI (optionnelle) dans Settings → Secrets : `OPENAI_API_KEY = "sk-..."`

## Notes
- L'application est conçue pour être utilisée en français.
- Si OpenAI n'est pas configuré, le fallback génère des QCM factices utilisables pour la démo.
