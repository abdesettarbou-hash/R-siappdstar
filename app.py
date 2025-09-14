import streamlit as st
import json
import os
import time
import sqlite3
from datetime import datetime
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import random

# Optional OpenAI client (if installed and configured on deployment)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="QCM Résidanat - FR", layout="wide")
st.title("📚 QCM Résidanat — Interface (FR)")

# ---------- Config ----------
DATA_DIR = "."
QCM_BANK_FILE = os.path.join(DATA_DIR, "qcm_bank.json")
DB_FILE = os.path.join(DATA_DIR, "results.db")

# Modules pré-définis et chapitres (exemple) — tri alphabétique
MODULES = {
    "Biochimie": ["Métabolisme du glucose", "Enzymologie", "Bioénergétique"],
    "Cardiologie": ["Insuffisance cardiaque", "Arythmies", "Cardiopathies ischémiques"],
    "Pharmacologie": ["Antibiotiques", "Antihypertenseurs", "Anticoagulants"],
    "Physiologie": ["Homéostasie", "Fonction rénale", "Système nerveux autonome"]
}

# Ensure module lists are sorted alphabetically
for k in list(MODULES.keys()):
    MODULES[k] = sorted(MODULES[k])
MODULE_NAMES = sorted(list(MODULES.keys()))

# ---------- Helpers: OpenAI client ----------
def get_openai_client():
    if not OPENAI_AVAILABLE:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

client = get_openai_client()
if OPENAI_AVAILABLE and client is None:
    st.sidebar.warning("OpenAI non configuré — la génération AI est désactivée. Ajoutez OPENAI_API_KEY aux secrets.")

# ---------- DB ----------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS examens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        module TEXT,
        sujet TEXT,
        mode TEXT,
        score INTEGER,
        total INTEGER
    )
    ''')
    conn.commit()
    conn.close()

def save_result(module, sujet, mode, score, total):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO examens (date, module, sujet, mode, score, total) VALUES (?, ?, ?, ?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M"), module, sujet, mode, score, total))
    conn.commit()
    conn.close()

def load_results():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT date, module, sujet, mode, score, total FROM examens ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return rows

init_db()

# ---------- QCM bank ----------
def load_qcm_bank():
    if not os.path.exists(QCM_BANK_FILE):
        return []
    with open(QCM_BANK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_qcm_bank(qcms):
    with open(QCM_BANK_FILE, "w", encoding="utf-8") as f:
        json.dump(qcms, f, ensure_ascii=False, indent=2)

if not os.path.exists(QCM_BANK_FILE):
    save_qcm_bank([])

# ---------- AI Generator (strict JSON if used) ----------
def generate_qcm_via_ai(module, sujet, n):
    if client is None:
        return None, "OpenAI non configuré"
    prompt = f"""Génère exactement {n} QCM au format JSON (array) en français pour le concours de résidanat.
Chaque item doit contenir:
- question: string
- choices: object avec clés 'A','B','C','D','E'
- correct_answer: 'A'|'B'|'C'|'D'|'E'
- explanation: string (courte)
Le thème: {sujet}; module: {module}.
Réponds uniquement avec le JSON, rien d'autre.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"Tu es un créateur de QCM médical, réponds uniquement en français."},
                      {"role":"user","content":prompt}],
            temperature=0.2
        )
        content = resp.choices[0].message.content.strip()
        # strip ``` fences
        if content.startswith("```"):
            content = "\n".join([line for line in content.splitlines() if not line.startswith("```")])
        data = json.loads(content)
        return data, None
    except Exception as e:
        return None, str(e)

# ---------- Fallback deterministic generator ----------
def generate_qcm_fallback(module, sujet, n):
    qcms = []
    for i in range(n):
        choices = {
            "A": f"Proposition A (réponse plausible)",
            "B": f"Proposition B",
            "C": f"Proposition C",
            "D": f"Proposition D",
            "E": f"Proposition E"
        }
        correct = random.choice(list(choices.keys()))
        qcms.append({
            "id": int(time.time()*1000) + random.randint(0,999),
            "module": module,
            "sujet": sujet,
            "question": f"Question factice {i+1} sur {sujet}",
            "choices": choices,
            "correct_answer": correct,
            "explanation": "Explication factice. Remplacer par contenu pédagogique."
        })
    return qcms

# ---------- UI layout: tabs ----------
tabs = st.tabs(["Générer QCM", "QCM interactif", "Examen blanc", "Bank & Dashboard"])

# --- Tab: Générer QCM ---
with tabs[0]:
    st.header("Générer des QCM (FR)")
    col1, col2 = st.columns([2,1])
    with col1:
        module_sel = st.selectbox("Module", MODULE_NAMES)
        chapitres = MODULES.get(module_sel, [])
        chapitre_sel = st.selectbox("Chapitre / Sujet", chapitres)
        n_qcm = st.slider("Nombre de QCM à générer", 1, 30, 5)
        use_ai = st.checkbox("Utiliser OpenAI si disponible", value=False)
    with col2:
        if st.button("Générer & Prévisualiser"):
            if use_ai and client:
                data, err = generate_qcm_via_ai(module_sel, chapitre_sel, n_qcm)
                if err:
                    st.error("Erreur OpenAI: " + err)
                    st.info("Utilisation du générateur de secours.")
                    data = generate_qcm_fallback(module_sel, chapitre_sel, n_qcm)
            else:
                data = generate_qcm_fallback(module_sel, chapitre_sel, n_qcm)
            # show preview and allow saving to bank
            st.success(f"{len(data)} QCM générés")
            for i,item in enumerate(data, start=1):
                st.markdown(f"**Q{i}: {item.get('question','')}**")
                ch = item.get("choices", {})
                for k in ['A','B','C','D','E']:
                    if k in ch:
                        st.write(f"{k}. {ch[k]}")
                st.info(f"Réponse correcte: {item.get('correct_answer','-')} — Explication: {item.get('explanation','')}")
            if st.button("Ajouter ces QCM au bank"):
                bank = load_qcm_bank()
                for item in data:
                    bank.append(item)
                save_qcm_bank(bank)
                st.success("QCM ajoutés au bank.")

# --- Tab: QCM interactif ---
with tabs[1]:
    st.header("QCM interactif — pratique")
    bank = load_qcm_bank()
    if not bank:
        st.info("Le bank est vide. Générez ou importez des QCM dans l'onglet 'Générer QCM'.")
    else:
        # select module and sujet from bank (alphabetical)
        modules_in_bank = sorted(list({b['module'] for b in bank}))
        sel_mod = st.selectbox("Module (bank)", modules_in_bank)
        sujets_in_mod = sorted(list({b['sujet'] for b in bank if b['module']==sel_mod}))
        sel_sujet = st.selectbox("Sujet (bank)", sujets_in_mod)
        # filter questions
        pool = [b for b in bank if b['module']==sel_mod and b['sujet']==sel_sujet]
        if not pool:
            st.warning("Pas de QCM pour ce module/sujet.")
        else:
            num = st.number_input("Nombre de questions à pratiquer", 1, min(30,len(pool)), value=min(10,len(pool)))
            if st.button("Démarrer séance"):
                session_q = []
                random.shuffle(pool)
                for idx,item in enumerate(pool[:num]):
                    session_q.append(item)
                st.session_state['interactive_qcms'] = session_q
                st.session_state['show_corrections'] = False
                st.session_state['interactive_answers'] = {}

            if st.session_state.get('interactive_qcms'):
                qlist = st.session_state['interactive_qcms']
                for i,q in enumerate(qlist, start=1):
                    st.markdown(f"**Q{i}: {q['question']}**")
                    choices = q.get('choices',{})
                    key = f"int_q_{q['id']}"
                    options = [k for k in ['A','B','C','D','E'] if k in choices]
                    ans = st.radio(f"Votre réponse Q{i}", options=options, key=key, format_func=lambda x, ch=choices: f"{x}. {ch.get(x,'')}" )
                    st.session_state['interactive_answers'][q['id']] = ans
                    # If corrections are revealed, show correct answer and explanation
                    if st.session_state.get('show_corrections', False):
                        corr = q.get('correct_answer')
                        if ans == corr:
                            st.success(f"Bonne réponse — {corr}")
                        else:
                            st.error(f"Mauvaise réponse — votre choix: {ans} | correcte: {corr}")
                        if q.get('explanation'):
                            st.info(q.get('explanation'))
                col_a, col_b = st.columns(2)
                if col_a.button("Afficher la correction"):
                    st.session_state['show_corrections'] = True
                if col_b.button("Effacer la session"):
                    st.session_state.pop('interactive_qcms', None)
                    st.session_state.pop('interactive_answers', None)
                    st.session_state.pop('show_corrections', None)
                    st.experimental_rerun()

# --- Tab: Examen blanc ---
with tabs[2]:
    st.header("Examen blanc — simulation chrono et correction à la fin")
    use_bank = st.radio("Source des questions", ("Bank local","Générer via AI / fallback"))
    if use_bank == "Bank local":
        bank = load_qcm_bank()
        modules_local = sorted(list({b['module'] for b in bank})) if bank else []
        sel_mod_exam = st.selectbox("Module (bank)", modules_local) if modules_local else None
        sujets_local = sorted(list({b['sujet'] for b in bank if b['module']==sel_mod_exam})) if sel_mod_exam else []
        sel_sujet_exam = st.selectbox("Sujet (bank)", sujets_local) if sujets_local else None
    else:
        sel_mod_exam = st.selectbox("Module (générer)", MODULE_NAMES)
        sel_sujet_exam = st.text_input("Sujet (générer)", value=MODULES.get(sel_mod_exam,["..."])[0])
    nb_exam = st.number_input("Nombre QCM (examen)", min_value=5, max_value=100, value=20)
    dur = st.number_input("Durée (minutes)", min_value=5, max_value=240, value=30)
    if st.button("Lancer examen"):
        # prepare qcms
        qcms = []
        if use_bank == "Bank local" and sel_mod_exam and sel_sujet_exam:
            pool = [b for b in load_qcm_bank() if b['module']==sel_mod_exam and b['sujet']==sel_sujet_exam]
            if not pool:
                st.error("Pas assez de questions dans le bank pour ce selection.")
            else:
                random.shuffle(pool)
                for i in range(nb_exam):
                    qcms.append(pool[i % len(pool)])
        else:
            # generate via AI or fallback
            if client:
                data, err = generate_qcm_via_ai(sel_mod_exam, sel_sujet_exam, nb_exam)
                if data is None:
                    st.warning("AI indisponible: utilisation du fallback.")
                    qcms = generate_qcm_fallback(sel_mod_exam, sel_sujet_exam, nb_exam)
                else:
                    # ensure id/module/sujet present
                    for it in data:
                        it['id'] = it.get('id', int(time.time()*1000)+random.randint(0,999))
                        it['module'] = it.get('module', sel_mod_exam)
                        it['sujet'] = it.get('sujet', sel_sujet_exam)
                        qcms.append(it)
            else:
                qcms = generate_qcm_fallback(sel_mod_exam, sel_sujet_exam, nb_exam)
        # store in session and render exam page
        st.session_state['exam_qcms'] = qcms
        st.session_state['exam_start'] = time.time()
        st.session_state['exam_duration'] = int(dur)*60
        st.session_state['exam_answers'] = {}
        st.session_state['exam_finished'] = False
        st.success("Examen démarré — bonne chance!")

    # render ongoing exam
    if st.session_state.get('exam_qcms') and not st.session_state.get('exam_finished', False):
        elapsed = time.time() - st.session_state.get('exam_start', 0)
        remaining = st.session_state.get('exam_duration', 0) - elapsed
        if remaining <= 0:
            st.session_state['exam_finished'] = True
            st.warning("Temps écoulé — correction automatique en cours.")
        else:
            mins = int(remaining//60); secs = int(remaining%60)
            st.info(f"Temps restant: {mins} min {secs} sec")
            for idx, q in enumerate(st.session_state['exam_qcms'], start=1):
                st.markdown(f"**Q{idx}: {q.get('question','')}**")
                choices = q.get('choices', {})
                options = [k for k in ['A','B','C','D','E'] if k in choices]
                ans_key = f"exam_q_{q['id']}"
                sel = st.radio(f"Votre réponse Q{idx}", options=options, key=ans_key,
                               format_func=lambda x, ch=choices: f"{x}. {ch.get(x,'')}" )
                st.session_state['exam_answers'][q['id']] = sel
            if st.button('Soumettre examen'):
                st.session_state['exam_finished'] = True

    # Correction and scoring
    if st.session_state.get('exam_finished', False) and st.session_state.get('exam_qcms'):
        qlist = st.session_state['exam_qcms']
        answers = st.session_state.get('exam_answers', {})
        score = 0
        details = []
        for i,q in enumerate(qlist, start=1):
            corr = q.get('correct_answer')
            chosen = answers.get(q['id'])
            is_cor = (chosen == corr)
            if is_cor:
                score += 1
            details.append({
                'index': i, 'question': q.get('question'), 'chosen': chosen, 'correct': corr, 'explanation': q.get('explanation','')
            })
        total = len(qlist)
        st.success(f"Score: {score} / {total} ({score/total*100:.1f} %)")
        for d in details:
            if d['chosen'] == d['correct']:
                st.markdown(f"✅ Q{d['index']}: {d['question']}")
            else:
                st.markdown(f"❌ Q{d['index']}: {d['question']}")
                st.write(f"Votre: {d['chosen']} — Correct: {d['correct']}")
            if d.get('explanation'):
                st.info(d['explanation'])
        # save result
        try:
            save_result(qlist[0].get('module','-'), qlist[0].get('sujet','-'), 'Examen blanc', score, total)
            st.success('Résultat sauvegardé.')
        except Exception as e:
            st.error('Erreur sauvegarde: ' + str(e))
        # cleanup session to allow new exam
        if st.button('Terminer et réinitialiser'):
            for k in ['exam_qcms','exam_start','exam_duration','exam_answers','exam_finished']:
                st.session_state.pop(k, None)
            st.experimental_rerun()

# --- Tab: Bank & Dashboard ---
with tabs[3]:
    st.header('Bank de QCM & Dashboard')
    bank = load_qcm_bank()
    st.write(f"Total QCM stockés: {len(bank)}")
    c1, c2 = st.columns([2,1])
    with c1:
        mod_filter = st.selectbox('Filtrer module (tous)', ['Tous'] + MODULE_NAMES)
        if mod_filter != 'Tous':
            sujet_choices = sorted(list({b['sujet'] for b in bank if b['module']==mod_filter}))
        else:
            sujet_choices = sorted(list({b['sujet'] for b in bank}))
        sujet_filter = st.selectbox('Filtrer sujet (tous)', ['Tous'] + (sujet_choices if sujet_choices else ['Tous']))
        # show filtered list
        filtered = bank
        if mod_filter != 'Tous':
            filtered = [b for b in filtered if b['module']==mod_filter]
        if sujet_filter != 'Tous':
            filtered = [b for b in filtered if b['sujet']==sujet_filter]
        for b in filtered[-200:]:  # show last 200
            st.markdown(f"**[{b.get('module')}] {b.get('sujet')}** — {b.get('question')}")
    with c2:
        if st.button('Télécharger bank (JSON)'):
            st.download_button('Télécharger qcm_bank.json', data=json.dumps(bank, ensure_ascii=False, indent=2),
                               file_name='qcm_bank.json', mime='application/json')
        if st.button('Vider bank'):
            save_qcm_bank([])
            st.experimental_rerun()
    # Dashboard: basic results
    rows = load_results()
    if rows:
        df = pd.DataFrame(rows, columns=['Date','Module','Sujet','Mode','Score','Total'])
        df['%'] = (df['Score']/df['Total'])*100
        df['Date'] = pd.to_datetime(df['Date'])
        st.subheader('Statistiques rapides')
        col1, col2, col3 = st.columns(3)
        col1.metric('Moyenne générale', f"{df['%'].mean():.1f} %")
        col2.metric('Dernier score', f"{df['%'].iloc[-1]:.1f} %")
        col3.metric('Meilleur score', f"{df['%'].max():.1f} %")
        # plot progression
        fig, ax = plt.subplots()
        df_sorted = df.sort_values('Date')
        ax.plot(df_sorted['Date'], df_sorted['%'], marker='o')
        ax.set_title('Progression des scores (%)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Score (%)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info('Aucun résultat enregistré.')
