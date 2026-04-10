import re
import PyPDF2
from sentence_transformers import SentenceTransformer, util

# ---------------------------
# LOAD BERT MODEL
# ---------------------------
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------
# SKILL DATABASE (CLEAN)
# ---------------------------
SKILL_DB = [
    # Programming
    "python", "java", "c++", "c", "javascript",

    # AI / Data
    "machine learning", "deep learning", "nlp", "data science",
    "tensorflow", "pytorch", "pandas", "numpy",

    # Database
    "sql", "mysql", "mongodb",

    # Tools
    "excel", "power bi", "tableau",

    # Systems
    "linux", "unix", "docker", "kubernetes",

    # Networking
    "networking", "tcp/ip", "routing", "switching",

    # Core Engineering
    "electronics", "circuit", "testing", "automation",
    "debugging", "troubleshooting"
]

# ---------------------------
# PDF TEXT EXTRACTION
# ---------------------------
def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# ---------------------------
# CLEAN TEXT
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------
# SKILL EXTRACTION (SMART)
# ---------------------------
def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill in SKILL_DB:
        if skill in text:
            found_skills.append(skill)

    return sorted(list(set(found_skills)))

# ---------------------------
# BERT SIMILARITY
# ---------------------------
def compute_bert_similarity(resumes, job_desc):

    job_emb = bert_model.encode(job_desc, convert_to_tensor=True)
    resume_emb = bert_model.encode(resumes, convert_to_tensor=True)

    scores = util.cos_sim(job_emb, resume_emb)[0]

    return scores.cpu().numpy()

# ---------------------------
# SKILL MATCHING
# ---------------------------
def skill_gap(candidate_skills, job_skills):
    matched = [s for s in job_skills if s in candidate_skills]
    missing = [s for s in job_skills if s not in candidate_skills]
    return matched, missing