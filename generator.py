import os
import json
import openai
from .prompts import build_qcm_prompt

def _safe_parse_json(text):
    # Attempt to extract JSON even if wrapped in backticks or markdown
    text = text.strip()
    # remove triple backticks if present
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.split('\n')[1:-1])
    # find first '['
    idx = text.find('[')
    if idx != -1:
        text = text[idx:]
    try:
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Impossible de parser la réponse IA en JSON: {e}\n---RESPONSE START---\n{text[:2000]}\n---RESPONSE END---")

def generate_qcms(subject, n=10, level="Moyen", include_explanations=True):
    api_key = os.getenv('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        # fallback example so app starts
        return [
            {
                "question": f"Exemple: Quel traitement initial pour {subject} ?",
                "choices": ["Option A","Option B","Option C","Option D"],
                "answer": "Option A",
                "explanation": "Réponse d'exemple, vérifier avec sources.",
                "source": "Exemple"
            }
            for _ in range(n)
        ]

    openai.api_key = api_key
    prompt = build_qcm_prompt(subject, n=n, level=level, include_explanations=include_explanations)

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=1500
    )
    text = response['choices'][0]['message']['content']
    data = _safe_parse_json(text)

    # basic validation
    out = []
    for item in data:
        q = item.get('question')
        choices = item.get('choices') or []
        answer = item.get('answer')
        if not (q and isinstance(choices, list) and len(choices) == 4 and answer in choices):
            raise ValueError('Format invalide reçu de l\'IA (question/4 choices/answer dans choices requis)')
        out.append({
            'question': q,
            'choices': choices,
            'answer': answer,
            'explanation': item.get('explanation',''),
            'source': item.get('source','')
        })
    return out
