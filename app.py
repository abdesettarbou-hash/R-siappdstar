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
import re

# Optional OpenAI client (if installed and configured)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="QCM Résidanat - FR", layout="wide")
st.title("📚 QCM Résidanat — Interface (FR)")

# ---------- Config & files ----------
DATA_DIR = "."
QCM_BANK_FILE = os.path.join(DATA_DIR, "qcm_bank.json")
DB_FILE = os.path.join(DATA_DIR, "results.db")

# ---------- Modules (larger list) ----------
MODULES = {
    "Anatomie": ["Appareil locomoteur", "Tête et cou", "Thorax", "Pelvis"],
    "Biochimie": ["Métabolisme du glucose", "Enzymologie", "Bioénergétique"],
    "Cardiologie": ["Insuffisance cardiaque", "Arythmies", "Cardiopathies ischémiques"],
    "Dermatologie": ["Dermatoses inflammatoires", "Infections cutanées"],
    "Endocrinologie": ["Diabète", "Thyroïde", "Surrénales"],
    "Gynécologie-Obstétrique": ["Grossesse", "Pathologies gynécologiques"],
    "Hématologie": ["Anémies", "Leucémies"],
    "Hépato-gastroentérologie": ["Cirrhose", "Pancréatite"],
    "Infectiologie / Microbiologie": ["Bactéries", "Virus", "Antibiothérapie"],
    "Médecine interne": ["Syndromes généraux", "Approche diagnostique"],
    "Néphrologie": ["Insuffisance rénale", "Troubles hydro-électrolytiques"],
    "Neurologie": ["Accident vasculaire cérébral", "Épilepsie"],
    "Oncologie": ["Tumeurs solides", "Chimiothérapie"],
    "Ophtalmologie": ["Réfraction", "Rétine"],
    "ORL": ["Otologie", "Rhinologie"],
    "Pédiatrie": ["Croissance", "Infections pédiatriques"],
    "Pharmacologie": ["Antibiotiques", "Antihypertenseurs", "Anticoagulants"],
    "Physiologie": ["Homéostasie", "Fonction rénale", "Système nerveux autonome"],
    "Psychiatrie": ["Dépression", "Psychoses"],
    "Pneumologie": ["BPCO", "Asthme"],
    "Rhumatologie": ["Polyarthrite", "Lupus"],
    "Urgences": ["Arrêt cardiaque", "Traumatologie"]
}

for k in list(MODULES.keys()):
    MODULES[k] = sorted(MODULES[k])
MODULE_NAMES = sorted(list(MODULES.keys()))

# ---------- OpenAI helper ----------
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

# ---------- DB helpers ----------
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
        try:
            return json.load(f)
        except Exception:
            return []

def save_qcm_bank(qcms):
    with open(QCM_BANK_FILE, "w", encoding="utf-8") as f:
        json.dump(qcms, f, ensure_ascii=False, indent=2)

if not os.path.exists(QCM_BANK_FILE):
    save_qcm_bank([])

# ---------- AI generator (requests JSON) ----------
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
        # strip code fences
        if content.startswith("```"):
            content = "\n".join([line for line in content.splitlines() if not line.startswith("```")])
        # try load json
        data = json.loads(content)
        if isinstance(data, list):
            # ensure keys exist for each item
            clean = []
            for it in data:
                if 'question' in it and 'choices' in it:
                    # fill missing fields
                    it.setdefault('correct_answer', None)
                    it.setdefault('explanation', '')
                    it.setdefault('id', int(time.time()*1000)+random.randint(0,999))
                    it.setdefault('module', module)
                    it.setdefault('sujet', sujet)
                    clean.append(it)
            return clean, None
        else:
            return None, "AI n'a pas retourné un array JSON."
    except Exception as e:
        return None, str(e)

# ---------- fallback parser: try to parse plain text QCM output ----------
def parse_qcm_text(content):
    # find questions of pattern Q1:, Q1 - etc.
    q_blocks = re.split(r"\nQ\d+[:\.\-]", "\n" + content)
    items = []
    if len(q_blocks) <= 1:
        # fallback: try split by lines starting with digit+.
        lines = content.splitlines()
        # give up
        return []
    # each block corresponds to a Q (skip first empty)
    for b in q_blocks[1:]:
        # get first line as question until newline
        parts = b.strip().splitlines()
        if not parts:
            continue
        qtext = parts[0].strip()
        choices = {}
        correct = None
        explanation = ''
        # scan following lines for choices and answer
        for line in parts[1:]:
            m = re.match(r"^\s*([A-E])[\)\.:\-]\s*(.*)$", line)
            if m:
                choices[m.group(1)] = m.group(2).strip()
            else:
                # check for Réponse or Réponse correcte
                rm = re.search(r"Réponse\s*(?:correcte|:)?\s*[:]?\s*([A-E])", line, re.IGNORECASE)
                if rm:
                    correct = rm.group(1).upper()
                # explanation
                if line.lower().startswith("explication") or line.lower().startswith("justification") or line.lower().startswith("raison"):
                    explanation += line + ' '
        if choices:
            items.append({
                'question': qtext,
                'choices': choices,
                'correct_answer': correct,
                'explanation': explanation,
                'id': int(time.time()*1000)+random.randint(0,999)
            })
    return items

# ---------- fallback deterministic generator ----------
def generate_qcm_fallback(module, sujet, n):
    qcms = []
    for i in range(n):
        choices = {
            "A": f"Proposition A (plausible)",
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

# ---------- UI: tabs ----------
tabs = st.tabs(["Générer QCM", "QCM interactif", "Examen blanc", "Bank & Dashboard"])

# --- Generate tab ---
with tabs[0]:
    st.header("Générer des QCM (FR)")
    col1, col2 = st.columns([2,1])
    with col1:
        module_sel = st.selectbox("Module", MODULE_NAMES)
        # allow custom module
        if st.checkbox("Autre module (saisir manuellement)", value=False):
            module_sel = st.text_input("Saisir module personnalisé", value=module_sel)

        chapitres = MODULES.get(module_sel, []) if module_sel in MODULES else []
        chapitre_sel = None
        if chapitres:
            chapitre_sel = st.selectbox("Chapitre / Sujet", chapitres)
        else:
            chapitre_sel = st.text_input("Chapitre / Sujet (saisir)", value="Général")
        if st.checkbox("Autre chapitre (saisir manuellement)", value=False, key='chp_custom'):
            chapitre_sel = st.text_input("Saisir chapitre personnalisé", value=chapitre_sel)

        n_qcm = st.number_input("Nombre de QCM à générer", min_value=1, max_value=100, value=10)
        use_ai = st.checkbox("Utiliser OpenAI si disponible", value=True)
    with col2:
        if st.button("Générer & Prévisualiser"):
            if use_ai and client:
                data, err = generate_qcm_via_ai(module_sel, chapitre_sel, n_qcm)
                if err or not data:
                    st.warning("Problème AI: " + str(err))
                    # try to parse raw AI text if available
                    # fallback to deterministic
                    data = generate_qcm_fallback(module_sel, chapitre_sel, n_qcm)
            else:
                data = generate_qcm_fallback(module_sel, chapitre_sel, n_qcm)

            # save temporary in session for interactive
            st.session_state['last_generated_qcms'] = data
            st.success(f"{len(data)} QCM générés (prévisualisation)")
            for i,item in enumerate(data, start=1):
                st.markdown(f"**Q{i}: {item.get('question','')}**")
                ch = item.get('choices', {})
                for k in ['A','B','C','D','E']:
                    if k in ch:
                        st.write(f"{k}. {ch[k]}")
                # hide correct answer initially in preview but show if user wants
                if st.checkbox(f"Afficher correction Q{i}", key=f"show_prev_corr_{i}"):
                    st.info(f"Réponse correcte: {item.get('correct_answer','-')} — Explication: {item.get('explanation','')}")

            if st.button("Ajouter ces QCM au bank (persistant)", key='add_bank'):
                bank = load_qcm_bank()
                for it in data:
                    bank.append(it)
                save_qcm_bank(bank)
                st.success("QCM ajoutés au bank.")

# --- Interactive tab ---
with tabs[1]:
    st.header("QCM interactif — pratique")
    bank = load_qcm_bank()
    if not bank:
        st.info("Le bank est vide. Générez ou importez des QCM via l'onglet Générer.")
    else:
        modules_in_bank = sorted(list({b['module'] for b in bank}))
        sel_mod = st.selectbox("Module (bank)", modules_in_bank)
        sujets_in_mod = sorted(list({b['sujet'] for b in bank if b['module']==sel_mod}))
        sel_sujet = st.selectbox("Sujet (bank)", sujets_in_mod if sujets_in_mod else ['Général'])
        pool = [b for b in bank if b['module']==sel_mod and b['sujet']==sel_sujet]
        if not pool:
            st.warning("Pas de QCM pour ce module/sujet.")
        else:
            num = st.number_input("Nombre de questions à pratiquer", min_value=1, max_value=min(50,len(pool)), value=min(10,len(pool)))
            if st.button("Démarrer séance interactive"):
                random.shuffle(pool)
                session_q = pool[:num]
                st.session_state['interactive_qcms'] = session_q
                st.session_state['interactive_answers'] = {}
                st.session_state['interactive_show_corrections'] = False
            if st.session_state.get('interactive_qcms'):
                qlist = st.session_state['interactive_qcms']
                for i,q in enumerate(qlist, start=1):
                    st.markdown(f"**Q{i}: {q['question']}**")
                    choices = q.get('choices',{})
                    options = [k for k in ['A','B','C','D','E'] if k in choices]
                    key = f"int_q_{q['id']}"
                    ans = st.radio(f"Votre réponse Q{i}", options=options, key=key, format_func=lambda x, ch=choices: f"{x}. {ch.get(x,'')}")
                    st.session_state['interactive_answers'][q['id']] = ans
                    if st.session_state.get('interactive_show_corrections', False):
                        corr = q.get('correct_answer')
                        if ans == corr:
                            st.success(f"Bonne réponse — {corr}")
                        else:
                            st.error(f"Mauvaise réponse — votre choix: {ans} | correcte: {corr}")
                        if q.get('explanation'):
                            st.info(q.get('explanation'))
                c1, c2 = st.columns(2)
                if c1.button('Afficher la correction (séance)'):
                    st.session_state['interactive_show_corrections'] = True
                if c2.button('Terminer séance'):
                    for k in ['interactive_qcms','interactive_answers','interactive_show_corrections']:
                        st.session_state.pop(k, None)
                    st.experimental_rerun()

# --- Exam tab ---
with tabs[2]:
    st.header('Examen blanc — simulation chrono et correction à la fin')
    src = st.radio('Source des questions', ('Bank local','Générer via AI / fallback'))
    if src == 'Bank local':
        bank = load_qcm_bank()
        modules_local = sorted(list({b['module'] for b in bank})) if bank else []
        sel_mod_exam = st.selectbox('Module (bank)', modules_local) if modules_local else None
        sujets_local = sorted(list({b['sujet'] for b in bank if b['module']==sel_mod_exam})) if sel_mod_exam else []
        sel_sujet_exam = st.selectbox('Sujet (bank)', sujets_local) if sujets_local else None
    else:
        sel_mod_exam = st.selectbox('Module (générer)', MODULE_NAMES)
        sel_sujet_exam = st.text_input('Sujet (générer)', value=MODULES.get(sel_mod_exam,["..."])[0])
    nb_exam = st.number_input('Nombre QCM (examen)', min_value=5, max_value=200, value=20)
    dur = st.number_input('Durée (minutes)', min_value=5, max_value=720, value=30)
    if st.button('Lancer examen blanc'):
        qcms = []
        if src == 'Bank local' and sel_mod_exam and sel_sujet_exam:
            pool = [b for b in load_qcm_bank() if b['module']==sel_mod_exam and b['sujet']==sel_sujet_exam]
            if not pool:
                st.error('Pas assez de questions dans le bank pour cette sélection.')
            else:
                random.shuffle(pool)
                for i in range(nb_exam):
                    qcms.append(pool[i % len(pool)])
        else:
            if client:
                data, err = generate_qcm_via_ai(sel_mod_exam, sel_sujet_exam, nb_exam)
                if data is None:
                    st.warning('AI indisponible ou erreur: utilisation du fallback.')
                    qcms = generate_qcm_fallback(sel_mod_exam, sel_sujet_exam, nb_exam)
                else:
                    qcms = data
            else:
                qcms = generate_qcm_fallback(sel_mod_exam, sel_sujet_exam, nb_exam)
        # store exam
        st.session_state['exam_qcms'] = qcms
        st.session_state['exam_start'] = time.time()
        st.session_state['exam_duration'] = int(dur) * 60
        st.session_state['exam_answers'] = {}
        st.session_state['exam_finished'] = False
        st.success('Examen lancé.')

    # render exam ongoing
    if st.session_state.get('exam_qcms') and not st.session_state.get('exam_finished', False):
        elapsed = time.time() - st.session_state.get('exam_start', 0)
        remaining = st.session_state.get('exam_duration', 0) - elapsed
        if remaining <= 0:
            st.session_state['exam_finished'] = True
            st.warning('Temps écoulé — correction automatique en cours.')
        else:
            m = int(remaining // 60); s = int(remaining % 60)
            st.info(f'Temps restant: {m} min {s} sec')
            for idx, q in enumerate(st.session_state['exam_qcms'], start=1):
                st.markdown(f"**Q{idx}: {q.get('question')}**")
                choices = q.get('choices', {})
                options = [k for k in ['A','B','C','D','E'] if k in choices]
                key = f"exam_q_{q['id']}"
                sel = st.radio(f'Votre réponse Q{idx}', options=options, key=key, format_func=lambda x, ch=choices: f"{x}. {ch.get(x,'')}")
                st.session_state['exam_answers'][q['id']] = sel
            if st.button('Soumettre examen'):
                st.session_state['exam_finished'] = True

    # scoring
    if st.session_state.get('exam_finished', False) and st.session_state.get('exam_qcms'):
        qlist = st.session_state['exam_qcms']
        answers = st.session_state.get('exam_answers', {})
        score = 0; details = []
        for i,q in enumerate(qlist, start=1):
            corr = q.get('correct_answer')
            chosen = answers.get(q['id'])
            is_cor = (chosen == corr)
            if is_cor:
                score += 1
            details.append({'index': i, 'question': q.get('question'), 'chosen': chosen, 'correct': corr, 'explanation': q.get('explanation','')})
        total = len(qlist)
        st.success(f'Score: {score} / {total} ({score/total*100:.1f} %)')
        for d in details:
            if d['chosen'] == d['correct']:
                st.markdown(f"✅ Q{d['index']}: {d['question']}")
            else:
                st.markdown(f"❌ Q{d['index']}: {d['question']}")
                st.write(f"Votre: {d['chosen']} — Correct: {d['correct']}")
            if d.get('explanation'):
                st.info(d['explanation'])
        # save
        try:
            save_result(qlist[0].get('module','-'), qlist[0].get('sujet','-'), 'Examen blanc', score, total)
            st.success('Résultat sauvegardé.')
        except Exception as e:
            st.error('Erreur sauvegarde: ' + str(e))
        if st.button('Terminer et réinitialiser examen'):
            for k in ['exam_qcms','exam_start','exam_duration','exam_answers','exam_finished']:
                st.session_state.pop(k, None)
            st.experimental_rerun()

# --- Bank & Dashboard ---
with tabs[3]:
    st.header('Bank de QCM & Dashboard')
    bank = load_qcm_bank()
    st.write(f'Total QCM stockés: {len(bank)}')
    c1, c2 = st.columns([2,1])
    with c1:
        mod_filter = st.selectbox('Filtrer module', ['Tous'] + MODULE_NAMES)
        if mod_filter != 'Tous':
            sujet_choices = sorted(list({b['sujet'] for b in bank if b['module']==mod_filter}))
        else:
            sujet_choices = sorted(list({b['sujet'] for b in bank}))
        sujet_filter = st.selectbox('Filtrer sujet', ['Tous'] + (sujet_choices if sujet_choices else ['Tous']))
        filtered = bank
        if mod_filter != 'Tous':
            filtered = [b for b in filtered if b['module']==mod_filter]
        if sujet_filter != 'Tous':
            filtered = [b for b in filtered if b['sujet']==sujet_filter]
        for b in filtered[-200:]:
            st.markdown(f"**[{b.get('module')}] {b.get('sujet')}** — {b.get('question')}")
    with c2:
        if st.button('Télécharger bank (JSON)'):
            st.download_button('Télécharger qcm_bank.json', data=json.dumps(bank, ensure_ascii=False, indent=2),
                               file_name='qcm_bank.json', mime='application/json')
        if st.button('Vider bank'):
            save_qcm_bank([])
            st.experimental_rerun()
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

if __name__ == '__main__':
    pass
