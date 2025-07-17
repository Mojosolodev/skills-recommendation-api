from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")
print("Model loaded.")

# Classe de la requête
class JobTitle(BaseModel):
    job_title: str

# Liste enrichie de soft skills + langues naturelles + langages de programmation
SOFT_SKILLS = [
    # Soft skills
    "communication", "teamwork", "creativity", "problem solving", "adaptability",
    "time management", "leadership", "empathy", "critical thinking", "attention to detail",
    "public speaking", "decision making", "negotiation", "collaboration", "patience",
    "multitasking", "persuasion", "self-motivation", "organization", "open-mindedness",
    "integrity", "active listening", "planning", "delegation", "independence",
    "customer focus", "confidence", "storytelling", "goal setting", "stress management",
    "strategic thinking", "emotional intelligence", "resourcefulness", "initiative",
    "flexibility", "analytical reasoning", "instructional skills", "mentoring",
    "coaching", "data literacy", "presentation skills", "networking", "research skills",

    # Human languages
    "english", "french", "spanish", "german", "arabic", "chinese", "russian",
    "portuguese", "italian", "japanese", "korean", "dutch", "swedish", "polish",

    # Programming languages
    "python", "javascript", "java", "c++", "c#", "typescript", "php", "ruby",
    "swift", "go", "kotlin", "html", "css", "sql", "bash"
]

# Embedding helper
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Fonction principale de recommandation
def recommend_soft_skills(job_title, top_k=5):
    job_emb = get_embedding(job_title)
    skill_embs = [get_embedding(skill) for skill in SOFT_SKILLS]
    scores = [cosine_similarity(job_emb, s_emb)[0][0] for s_emb in skill_embs]
    sorted_skills = sorted(zip(SOFT_SKILLS, scores), key=lambda x: x[1], reverse=True)
    return [skill for skill, score in sorted_skills[:top_k]]

# Endpoint API
@app.post("/recommend")
def recommend(data: JobTitle):
    print(f"Job title received: {data.job_title}")
    skills = recommend_soft_skills(data.job_title)
    print(f"Recommended skills: {skills}")
    return {"skills": skills}
