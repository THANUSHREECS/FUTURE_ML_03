import streamlit as st
import pandas as pd
from utils import *
from PIL import Image

st.set_page_config(page_title="AI Resume Screening", layout="wide")

# ---------------------------
# TITLE + BANNER
# ---------------------------
st.title("🤖 AI Resume Screening System")

image = Image.open("banner.png")
st.image(image, use_container_width=True)

# ---------------------------
# INPUT
# ---------------------------
job_description = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# ---------------------------
# PROCESS
# ---------------------------
if uploaded_files and job_description:

    resumes = []
    names = []

    for file in uploaded_files:
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
        else:
            text = file.read().decode("utf-8")

        text = clean_text(text)
        resumes.append(text)
        names.append(file.name)

    # Clean job description
    job_clean = clean_text(job_description)

    # 🔥 Extract required skills from job
    job_skills = extract_skills(job_clean)

    # 🔥 BERT scoring
    scores = compute_bert_similarity(resumes, job_clean)
    scores = [round(float(s) * 100, 2) for s in scores]

    data = []

    for i in range(len(resumes)):
        candidate_skills = extract_skills(resumes[i])

        matched, missing = skill_gap(candidate_skills, job_skills)

        data.append({
            "Candidate": names[i],
            "Score": scores[i],
            "Skills": candidate_skills,
            "Matched Skills": matched,
            "Missing Skills": missing
        })

    df = pd.DataFrame(data)
    df = df.sort_values(by="Score", ascending=False)

    # ---------------------------
    # OUTPUT
    # ---------------------------
    st.subheader("🏆 Ranked Candidates")
    st.dataframe(df, use_container_width=True)

    st.success(
        f"Top Candidate: {df.iloc[0]['Candidate']} (Score: {df.iloc[0]['Score']})"
    )

    # ---------------------------
    # CHART
    # ---------------------------
    st.subheader("📊 Score Chart")
    st.bar_chart(df.set_index("Candidate")["Score"])

    # ---------------------------
    # DOWNLOAD
    # ---------------------------
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Results",
        data=csv,
        file_name="results.csv",
        mime="text/csv"
    )