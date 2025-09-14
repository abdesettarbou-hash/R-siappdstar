
import os
import streamlit as st
import sqlite3
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Optional OpenAI import - will fail if not configured; we handle gracefully
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Initialize OpenAI client if possible (expects OPENAI_API_KEY env var)
def get_openai_client():
    if not OPENAI_AVAILABLE:
        return None
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# ===== DB helpers =====
DB_PATH = "results.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS examens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        module TEXT,
        sujet TEXT,
        mode TEXT,
        score INTEGER,
        total INTEGER
    )
    """)
    conn.commit()
    conn.close()

def save_result(module, sujet, mode, score, total):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO examens (date, module, sujet, mode, score, total) VALUES (?, ?, ?, ?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M"), module, sujet, mode, score, total))
    conn.commit()
    conn.close()

def load_results():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT date, module, sujet, mode, score, total FROM examens ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return rows

# Initialize DB on startup
init_db()

# ===== Streamlit UI =====
st.set_page_config(page_title="Générateur & Simulateur QCM - Résidanat", layout="wide")
st.title("🎓 Générateur & Simulateur QCM — Résidanat (Prototype)")

client = get_openai_client()
if client is None:
    st.warning("OpenAI client not configured. Set environment variable OPENAI_API_KEY or configure Streamlit secrets to enable AI features (génération, correction, résumé). The app will still run offline for dashboard features.")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Simulation QCM", "Dashboard", "Instructions"])

# ---------------- Simulation QCM ----------------
if page == "Simulation QCM":
    st.header("Simulation QCM / Génération de questions")
    module = st.selectbox("Module", ["Biochimie", "Pharmacologie", "Physiologie", "Pathologie"])
    sujet = st.text_input("Sujet (ex: Métabolisme du glucose)", value="Métabolisme du glucose")
    nb_qcm = st.slider("Nombre de QCM", 1, 20, 5)
    duration = st.slider("Durée de l'examen (minutes)", 1, 60, 10)
    mode = st.radio("Mode", ["Révision (corrigé immédiat)", "Examen (corrigé à la fin)"])

    if st.button("🚀 Générer & Lancer Simulation"):
        with st.spinner("Génération en cours (si OpenAI configuré)..."):
            # If OpenAI configured, ask it to generate questions; otherwise use placeholders
            if client:
                prompt = f\"\"\"Génère {nb_qcm} QCM pour un concours de résidanat en {module} sur le thème \"{sujet}\".
Contraintes :
- Chaque QCM contient 1 question + 5 propositions de réponse (A à E).
- Une seule bonne réponse.
- Donne uniquement les questions + choix, sans correction dans le texte.
- Niveau : concours de résidanat.
Format clair en texte.
\"\"\"
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": f"Tu es un expert en {module} médicale."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2
                    )
                    output_text = response.choices[0].message.content
                except Exception as e:
                    st.error(f"Erreur OpenAI: {e}")
                    output_text = ""
            else:
                output_text = ""

            # Fallback simple generator if no AI
            qcm_list = []
            if output_text:
                # Try simple parsing assuming "Q1:" format
                for block in output_text.split("Q")[1:]:
                    lines = block.strip().split("\n")
                    if not lines:
                        continue
                    question = lines[0].split(": ", 1)[-1].strip()
                    choices = {}
                    for line in lines[1:]:
                        if line.startswith(("A.", "B.", "C.", "D.", "E.")):
                            letter, answer = line.split(".", 1)
                            choices[letter.strip()] = answer.strip()
                    qcm_list.append({"question": question, "choices": choices})
            else:
                # synthetic fallback QCM
                for i in range(1, nb_qcm+1):
                    q = f"Question factice {i} sur {sujet}"
                    choices = {"A": "Réponse A", "B": "Réponse B", "C": "Réponse C", "D": "Réponse D", "E": "Réponse E"}
                    qcm_list.append({"question": q, "choices": choices})

            # Store in session
            st.session_state.qcm_list = qcm_list
            st.session_state.start_time = time.time()
            st.session_state.duration = duration * 60
            st.session_state.answers = {}
            st.session_state.mode = mode
            st.session_state.module = module
            st.session_state.sujet = sujet
            st.session_state.finished = False

# Display exam if generated
if page == "Simulation QCM" and "qcm_list" in st.session_state and not st.session_state.finished:
    elapsed = time.time() - st.session_state.start_time
    remaining = st.session_state.duration - elapsed
    if remaining <= 0:
        st.session_state.finished = True
        st.warning("⏰ Temps écoulé !")
    else:
        st.info(f"Temps restant : {int(remaining//60)} min {int(remaining%60)} sec")
        for i, qcm in enumerate(st.session_state.qcm_list, 1):
            st.markdown(f"**Q{i}: {qcm['question']}**")
            st.session_state.answers[i] = st.radio(f"Votre réponse Q{i}", list(qcm['choices'].keys()), key=f"q{i}")
            if st.session_state.mode.startswith("Révision") and client:
                # immediate correction via OpenAI
                correction_prompt = f\"\"\"Question: {qcm['question']}
Choix:
{json.dumps(qcm['choices'], ensure_ascii=False, indent=2)}
Réponse de l'étudiant: {st.session_state.answers[i]}
Corrige la question et explique.\"\"\"
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"system","content":"Corrige QCM médical."},
                                  {"role":"user","content":correction_prompt}],
                        temperature=0
                    )
                    st.info(resp.choices[0].message.content)
                except Exception as e:
                    st.info("Correction automatique indisponible.")
        if st.session_state.mode.startswith("Examen"):
            if st.button("📤 Soumettre"):
                st.session_state.finished = True

# Correction and scoring
if page == "Simulation QCM" and "finished" in st.session_state and st.session_state.finished:
    st.success("Correction et résultats")
    # If OpenAI available, ask it to correct and provide explanations and count score
    if client:
        correction_prompt = "Corrige ces QCM avec réponses correctes + explications et calcule le score:\n"
        for i, qcm in enumerate(st.session_state.qcm_list, 1):
            correction_prompt += f"Q{i}: {qcm['question']}\n"
            for k, v in qcm["choices"].items():
                correction_prompt += f"{k}. {v}\n"
            correction_prompt += f"Réponse de l'étudiant: {st.session_state.answers.get(i, '-')}\n\n"

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"Tu es un correcteur de QCM médical."},
                          {"role":"user","content":correction_prompt}],
                temperature=0
            )
            result_text = response.choices[0].message.content
            st.write(result_text)
            # Note: automatic score parsing is not implemented robustly here.
            # For the prototype, you can manually extract the score or extend parsing logic.
        except Exception as e:
            st.error(f"Erreur correction OpenAI: {e}")
            st.write("Impossible de corriger automatiquement.")
            st.write("نتيجة مبدئية: 0 /", len(st.session_state.qcm_list))
    else:
        st.info("OpenAI non configuré — correction automatique indisponible.")
        st.write("نتيجة مبدئية: 0 /", len(st.session_state.qcm_list))

    # Save a placeholder result (score parsing not implemented)
    try:
        save_result(st.session_state.module, st.session_state.sujet, st.session_state.mode, 0, len(st.session_state.qcm_list))
        st.success("النتيجة سُجلت في قاعدة البيانات (score placeholder = 0).")
    except Exception as e:
        st.error(f"Erreur sauvegarde: {e}")

# ---------------- Dashboard ----------------
if page == "Dashboard":
    st.header("📊 Dashboard - Historique & Analyses")
    rows = load_results()
    if not rows:
        st.info("Aucun examen enregistré encore. قم بتشغيل محاكاة وحفظ نتيجة لتجربة الـDashboard.")
    else:
        df = pd.DataFrame(rows, columns=["Date", "Module", "Sujet", "Mode", "Score", "Total"])
        df["%"] = (df["Score"] / df["Total"]) * 100
        df["Date"] = pd.to_datetime(df["Date"])

        # Filters
        modules = ["Tous"] + sorted(df["Module"].unique().tolist())
        selected_module = st.sidebar.selectbox("📚 Filtrer par module", modules)
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()
        start_date, end_date = st.sidebar.date_input("📅 Filtrer par période", value=[min_date, max_date], min_value=min_date, max_value=max_date)

        # Comparison selectors
        all_modules = sorted(df["Module"].unique().tolist())
        module1 = st.sidebar.selectbox("Module 1 (comparaison)", all_modules, index=0)
        module2 = st.sidebar.selectbox("Module 2 (comparaison)", all_modules, index=1 if len(all_modules)>1 else 0)

        # Apply filters
        filtered_df = df.copy()
        if selected_module != "Tous":
            filtered_df = filtered_df[filtered_df["Module"] == selected_module]
        filtered_df = filtered_df[(filtered_df["Date"].dt.date >= start_date) & (filtered_df["Date"].dt.date <= end_date)]

        if filtered_df.empty:
            st.warning("⚠️ Aucun résultat pour ces filtres.")
        else:
            # Quick stats cards
            moyenne = filtered_df["%"].mean()
            dernier = filtered_df["%"].iloc[-1]
            meilleur = filtered_df["%"].max()
            col1, col2, col3 = st.columns(3)
            col1.metric("📈 Moyenne générale", f"{moyenne:.1f} %")
            col2.metric("🕒 Dernier score", f"{dernier:.1f} %")
            col3.metric("🏆 Meilleur score", f"{meilleur:.1f} %")

            # GPT summary (if available)
            summary_text = "Résumé automatique non disponible (OpenAI non configuré)."
            if client:
                prompt = f"Analyse ces résultats et fournis: moyenne générale (%), meilleur score et module, module le plus faible, recommandations. Résultats: {filtered_df.to_dict(orient='records')}"
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"system","content":"Tu es un expert pédagogique en médecine."},{"role":"user","content":prompt}],
                        temperature=0.4
                    )
                    summary_text = resp.choices[0].message.content
                except Exception:
                    summary_text = "Résumé automatique indisponible (erreur OpenAI)."
            st.subheader("Résumé analytique")
            st.write(summary_text)

            # Progression plot
            fig, ax = plt.subplots()
            ax.plot(filtered_df["Date"], filtered_df["%"], marker="o", linestyle="-")
            ax.set_title("Progression des scores (%)")
            ax.set_xlabel("Dates")
            ax.set_ylabel("Score (%)")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Module means bar chart
            st.subheader("📚 Moyenne par module")
            module_means = filtered_df.groupby("Module")["%"].mean().sort_values(ascending=False)
            fig2, ax2 = plt.subplots()
            module_means.plot(kind="bar", ax=ax2)
            ax2.set_ylabel("Score (%)")
            st.pyplot(fig2)

            # Comparison chart
            st.subheader(f"⚖️ Comparaison: {module1} vs {module2}")
            comp_df = df[df["Module"].isin([module1, module2])]
            comp_df = comp_df[(comp_df["Date"].dt.date >= start_date) & (comp_df["Date"].dt.date <= end_date)]
            if not comp_df.empty:
                fig3, ax3 = plt.subplots()
                for m in [module1, module2]:
                    sub = comp_df[comp_df["Module"] == m]
                    ax3.plot(sub["Date"], sub["%"], marker="o", linestyle="-", label=m)
                ax3.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig3)
            else:
                st.info("Pas assez de données pour la comparaison choisie.")

            # Heatmap
            st.subheader("🔥 Heatmap: Modules × Dates")
            pivot_df = filtered_df.pivot_table(index="Module", columns=filtered_df["Date"].dt.date, values="%", aggfunc="mean")
            if not pivot_df.empty:
                fig4, ax4 = plt.subplots(figsize=(10, 4))
                sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, ax=ax4)
                st.pyplot(fig4)
            else:
                st.info("Pas assez de données pour la heatmap.")

            # Forecast global
            st.subheader("🔮 Projection (forecast) globale")
            df_for_forecast = filtered_df.sort_values("Date")
            if len(df_for_forecast) > 2:
                df_for_forecast["days"] = (df_for_forecast["Date"] - df_for_forecast["Date"].min()).dt.days
                X = df_for_forecast[["days"]]
                y = df_for_forecast["%"]
                model = LinearRegression()
                model.fit(X, y)
                future_days = pd.np.arange(df_for_forecast["days"].max()+1, df_for_forecast["days"].max()+11).reshape(-1,1)
                y_pred = model.predict(future_days)
                future_dates = [df_for_forecast["Date"].max() + pd.Timedelta(days=i) for i in range(1,11)]
                fig5, ax5 = plt.subplots()
                ax5.plot(df_for_forecast["Date"], y, "bo-", label="Réel")
                ax5.plot(future_dates, y_pred, "r--", label="Projection")
                ax5.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig5)
            else:
                st.info("Pas assez de données pour la projection globale.")

            # Forecast by module
            st.subheader("🔮 Projection par module")
            module_forecast = st.selectbox("Choisir un module pour projection", df["Module"].unique())
            sub_df = df[df["Module"] == module_forecast].sort_values("Date")
            if len(sub_df) > 2:
                sub_df["days"] = (sub_df["Date"] - sub_df["Date"].min()).dt.days
                Xm = sub_df[["days"]]
                ym = sub_df["%"]
                model_m = LinearRegression()
                model_m.fit(Xm, ym)
                future_days_m = pd.np.arange(sub_df["days"].max()+1, sub_df["days"].max()+11).reshape(-1,1)
                y_pred_m = model_m.predict(future_days_m)
                future_dates_m = [sub_df["Date"].max() + pd.Timedelta(days=i) for i in range(1,11)]
                fig6, ax6 = plt.subplots()
                ax6.plot(sub_df["Date"], ym, "go-", label="Réel")
                ax6.plot(future_dates_m, y_pred_m, "r--", label="Projection")
                ax6.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig6)
            else:
                st.info("Pas assez de données pour la projection par module.")

            # Export filtered df to Excel
            excel_buf = BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                filtered_df.to_excel(writer, index=False, sheet_name="Résultats filtrés")
            st.download_button(label="📥 Exporter (Excel) - résultats filtrés", data=excel_buf, file_name="resultats_residanat_filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------- Instructions ----------------
if page == "Instructions":
    st.header("🔧 Instructions d'installation et d'exécution")
    st.markdown("""
**Prérequis**
- Python 3.9+ recommandé
- Installer dépendances: `pip install -r requirements.txt`

**Clés API**
- Pour activer la génération & correction via OpenAI, définissez la variable d'environnement `OPENAI_API_KEY`:
  - Linux/macOS: `export OPENAI_API_KEY='sk-...'`
  - Windows (PowerShell): `$env:OPENAI_API_KEY='sk-...'`
- Ou copiez la clé dans `.streamlit/secrets.toml`:
```
OPENAI_API_KEY = "sk-..."
```

**Lancer l'app**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Notes pratiques**
- Ce prototype utilise SQLite (fichier `results.db`) pour stocker l'historique.
- Les fonctionnalités de génération, correction et résumé nécessitent une connexion internet active et clé OpenAI valide.
- Pour la production: sécuriser la clé API, ajouter validation humaine des QCM, et mettre en place quotas / facturation.
""")

