# FUTURE_ML_03
AI-powered resume screening system using BERT and NLP

# 🤖 AI Resume Screening System

## Overview
This project is an AI-powered resume screening system that helps recruiters shortlist candidates based on job descriptions.

## Features
- Upload multiple resumes (PDF/TXT)
- Extract text using NLP
- Semantic matching using BERT
- Skill extraction using skill database
- Candidate ranking
- Skill gap analysis
- Download results as CSV

## Technologies Used
- Python
- Streamlit
- BERT (sentence-transformers)
- spaCy
- Scikit-learn

## How It Works
1. Extract text from resumes
2. Clean and preprocess text
3. Extract skills from job description
4. Compute similarity using BERT
5. Rank candidates
6. Identify matched & missing skills

## Output
- Ranked candidates with scores
- Extracted skills from resumes
- Matching skills with job description
- Missing skills analysis   

## Run the Project
pip install -r requirements.txt

streamlit run app.py
