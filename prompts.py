def build_qcm_prompt(subject, n=10, level="Moyen", include_explanations=True):
    template = f"""Tu es un professeur expert en {subject}. Fournis strictement un JSON (sans texte additionnel) : une liste d'objets.
Chaque objet doit contenir les champs :
- question : texte clair, pas plus de 2 phrases.
- choices : liste de 4 options (strings).
- answer : la chaîne exactement identique à l'un des éléments de choices.
- explanation : (optionnel) 1-2 phrases expliquant la réponse.
- source : (optionnel) brève référence (livre, guideline, chapitre).

Génère {n} questions de niveau {level}. Si possible, diversifie l'angle (diagnostic, prise en charge, pharmacologie).
Respecte ce format JSON strict :
[{{"question":"...","choices":["A","B","C","D"],"answer":"...","explanation":"...","source":"..."}}, ...]
Ne fournis rien d'autre qu'un JSON valide.
"""
    return template
