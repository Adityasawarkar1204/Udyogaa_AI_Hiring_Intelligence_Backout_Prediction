# src/matching.py

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"


def load_encoder():
    return SentenceTransformer(MODEL_NAME)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def match_job_resume(encoder, job_desc, resume_text):
    job_vec = encoder.encode(job_desc)
    resume_vec = encoder.encode(resume_text)
    return float(cosine_similarity(job_vec, resume_vec))

