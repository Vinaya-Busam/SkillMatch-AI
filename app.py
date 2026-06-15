import streamlit as st
import pickle
import pdfplumber
import re

# Load the pre-trained model
model = pickle.load(open('.pkl Files/resume_model.pkl', 'rb'))
vectorizer = pickle.load(open('.pkl Files/tfidf.pkl', 'rb'))

with open("skills.txt", "r") as f:
    SKILLS = [
        skill.strip().lower()
        for skill in f.readlines()
    ]

# Extract Resume Text 
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

# Extract Skills 
def extract_skills(text):
    text = text.lower()
    found_skills = []
    for skill in SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'

        if re.search(pattern, text):
            found_skills.append(skill)
    
    return sorted(list(set(found_skills)))

# ATS Score
def calculate_match_score(resume_skills, jd_skills):
    matched = set(resume_skills) & set(jd_skills)
    missing = set(jd_skills) - set(resume_skills)

    if len(jd_skills) == 0:
        score = 0
    else:
        score = round((len(matched) / len(jd_skills)) * 100, 2)
    
    return score, matched, missing



# Page Config

st.ste_page_config(
    page_title = "Resume Screening System",
    page_icon = "📄",
    layout = "wide"
)

# Header

st.title("Resume Screening System")

st.markdown(
    """
    Upload a resume and compare it with a Job Description.
    
    The system will:
    - Predict candidate category
    - Extract skills
    - Calculate ATS score
    - Show matched skills
    - Show missing skills
    """
)

# Inputs

col1, col2 = st.columns(2)

with col1:
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"]) 

with col2:
    job_description = st.text_area("Paste Job Description Here", height=250)


# Analysis
if uploaded_resume and job_description:
    resume_text = extract_text_from_pdf(uploaded_resume)

    resume_vector = vectorizer.transform([resume_text])
    predicted_category = model.predict(resume_vector)[0]

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description)

    # ATS Score
    score, matched, missing = calculate_match_score(resume_skills, jd_skills)

    st.divider()

    # Results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Predicted Category")
        st.success(predicted_category)
    
    with col2:
        st.subheader("ATS Match Score")
        st.progress(int(score))

        st.metric("Score", f"{score}%")

    st.divider()

    # Skills

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Resume Skills")

        if resume_skills:
            for skill in resume_skills:
                st.write(f"{skill}")
        else:
            st.warning("No skills detected")
        
    with col2:
        st.subheader("Matched Skills")

        if matched:
            for skill in sorted(matched):
                st.success(skill)
        else:
            st.warning("No matching skills")
    
    with col3:
        st.subheader("Missing Skills")

        if missing:
            for skill in sorted(missing):
                st.error(skill)
        else:
            st.success("No missing skills")

    st.divider()

    st.subheader("Resume Preview")

    st.text_area(
        "Extracted Resume Text",
        resume_text[:3000],
        height=250
    )

