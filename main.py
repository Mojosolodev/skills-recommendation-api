from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

print("Loading lightweight model...")

# Classe de la requête
class JobTitle(BaseModel):
    job_title: str

# Liste enrichie de soft skills + langues naturelles + langages de programmation
SOFT_SKILLS = [
    # Soft skills
    "communication", "teamwork", "creativity", "problem solving", "adaptability",
    "time management", "leadership", "empathy", "critical thinking",
    "public speaking", "decision making", "negotiation", "collaboration", "patience",
    "persuasive","integrity", "planning","confidence","storytelling","stress management",
    "strategic thinking", "emotional intelligence","initiative","flexible",
    "flexibility", "analysis","mentoring",
    "coaching","presentation skills", "networking", "research skills",

    # Human languages
    "english", "french", "spanish","chinese","italian", "japanese", "korean"

    # Programming languages
    "python", "javascript", "java", "c++", "c#", "typescript", "php", "ruby",
    "swift", "go", "kotlin", "html", "css", "sql", "bash","react native"
]

# === Simulation d'embeddings légers ===

# Dictionnaire statique simulant des vecteurs pour chaque soft skill
np.random.seed(42)
EMBEDDINGS = {skill: np.random.rand(10) for skill in SOFT_SKILLS}

# Fonction simulée de génération d'embedding pour un titre de poste
def get_fake_embedding(text: str) -> np.ndarray:
    # Hash simple basé sur le texte
    seed = sum(ord(c) for c in text.lower()) % 10000
    rng = np.random.default_rng(seed)
    return rng.random(10)

# Fonction principale de recommandation
def recommend_soft_skills(job_title, top_k=5):
    job_emb = get_fake_embedding(job_title)
    similarities = {
        skill: cosine_similarity([job_emb], [vec])[0][0]
        for skill, vec in EMBEDDINGS.items()
    }
    sorted_skills = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [skill for skill, _ in sorted_skills[:top_k]]

# Endpoint API
@app.post("/recommend")
def recommend(data: JobTitle):
    print(f"Job title received: {data.job_title}")
    skills = recommend_soft_skills(data.job_title)
    print(f"Recommended skills: {skills}")
    return {"skills": skills}
