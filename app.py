import streamlit as st
import json, os, time, random, sqlite3
from datetime import datetime
from openai import OpenAI

st.set_page_config(page_title="QCM Résidanat - FR", layout="wide")
st.title("📚 QCM Résidanat — Interface (FR)")

DATA_DIR = "."
QCM_BANK_FILE = os.path.join(DATA_DIR, "qcm_bank.json")
DB_FILE = os.path.join(DATA_DIR, "results.db")

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
for k in MODULES:
    MODULES[k] = sorted(MODULES[k])
MODULE_NAMES = sorted(list(MODULES.keys()))

try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    client = None
    st.error("OpenAI API key manquante ou mal configurée")

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
init_db()

def save_result(module, sujet, mode, score, total):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO examens (date,module,sujet,mode,score,total) VALUES (?,?,?,?,?,?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M"),module,sujet,mode,score,total))
    conn.commit()
    conn.close()

def load_results():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT date,module,sujet,mode,score,total FROM examens ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return rows

def load_qcm_bank():
    if not os.path.exists(QCM_BANK_FILE): return []
    with open(QCM_BANK_FILE,"r",encoding="utf-8") as f:
        try: return json.load(f)
        except: return []

def save_qcm_bank(qcms):
    with open(QCM_BANK_FILE,"w",encoding="utf-8") as f:
        json.dump(qcms,f,ensure_ascii=False,indent=2)

if not os.path.exists(QCM_BANK_FILE):
    save_qcm_bank([])

def generate_qcm_via_ai(module,sujet,n):
    if not client:
        return None,"OpenAI non configuré"
    prompt=f'''Generer exactement {n} QCM en francais pour le concours de residanat.
Chaque item doit contenir:
- question: string
- choices: objet avec cles 'A','B','C','D','E'
- correct_answer: 'A'-'E'
- explanation: string courte
Module: {module}, Sujet: {sujet}
Repond uniquement avec le JSON, rien d\'autre.'''
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"Tu es un createur de QCM medical, reponds uniquement en francais."},
                      {"role":"user","content":prompt}],
            temperature=0.2
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = "\n".join([line for line in content.splitlines() if not line.startswith("```")])
        data = json.loads(content)
        for it in data:
            it.setdefault('correct_answer',None)
            it.setdefault('explanation','')
            it.setdefault('id',int(time.time()*1000)+random.randint(0,999))
            it.setdefault('module',module)
            it.setdefault('sujet',sujet)
        return data,None
    except Exception as e:
        return None,str(e)

def generate_qcm_fallback(module,sujet,n):
    qcms = []
    for i in range(n):
        choices = {c:f"Proposition {c}" for c in "ABCDE"}
        correct = random.choice(list(choices.keys()))
        qcms.append({
            "id": int(time.time()*1000)+random.randint(0,999),
            "module": module,
            "sujet": sujet,
            "question": f"Question factice {i+1} sur {sujet}",
            "choices": choices,
            "correct_answer": correct,
            "explanation": "Explication factice"
        })
    return qcms

st.header("Générer des QCM")
module_sel = st.selectbox("Module", MODULE_NAMES)
chapitres = MODULES.get(module_sel,[])
chapitre_sel = st.selectbox("Chapitre / Sujet", chapitres) if chapitres else st.text_input("Chapitre / Sujet",value="Général")
n_qcm = st.number_input("Nombre de QCM à générer",1,100,10)
if st.button("Générer QCM"):
    data,err = generate_qcm_via_ai(module_sel,chapitre_sel,n_qcm) if client else (generate_qcm_fallback(module_sel,chapitre_sel,n_qcm),None)
    if err:
        st.warning(f"Problème AI: {err}. Fallback activé.")
        data = generate_qcm_fallback(module_sel,chapitre_sel,n_qcm)
    st.session_state['last_generated_qcms'] = data
    for i,item in enumerate(data,1):
        st.markdown(f"**Q{i}: {item['question']}**")
        for k,v in item['choices'].items():
            st.write(f"{k}. {v}")
        if st.checkbox(f"Afficher correction Q{i}",key=f"show_prev_{i}"):
            st.info(f"Réponse correcte: {item['correct_answer']} — Explication: {item['explanation']}")
    if st.button("Ajouter au bank"):
        bank = load_qcm_bank()
        bank.extend(data)
        save_qcm_bank(bank)
        st.success("QCM ajoutés au bank")

if __name__=="__main__": pass
