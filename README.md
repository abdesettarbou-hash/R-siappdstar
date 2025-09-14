# Application QCM Résidanat (FR)

Application Streamlit pour :
- Générer des QCM de résidanat en français via OpenAI
- Mode examen blanc
- Suivi et projection des résultats

## Installation locale
1. `pip install -r requirements.txt`
2. `streamlit run app.py`

## Déploiement (Streamlit Cloud)
- Hébergez ce repo sur [Streamlit Cloud](https://streamlit.io/cloud).
- Ajoutez votre clé OpenAI dans **Settings → Secrets** :

```toml
OPENAI_API_KEY="votre_clef"
```
