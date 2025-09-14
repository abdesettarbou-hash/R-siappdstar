# qcm/prompts.py
def build_qcm_prompt(subject, n=10, level="Moyen", include_explanations=True):
    return f"Prompt pour {n} QCM sur {subject} niveau {level}"
