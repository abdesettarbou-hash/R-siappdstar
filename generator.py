# qcm/generator.py
def generate_qcms(subject, n=10, level="Moyen", include_explanations=True):
    # Placeholder: fonction à compléter avec appel API OpenAI
    return [
        {
            "question": "Exemple de question sur " + subject,
            "choices": ["Option A", "Option B", "Option C", "Option D"],
            "answer": "Option A",
            "explanation": "Ceci est un exemple.",
            "source": "Source fictive"
        }
        for _ in range(n)
    ]
