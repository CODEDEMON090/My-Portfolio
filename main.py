# data_scientist_portfolio_streamlit.py
# Streamlit portfolio web app for a Data Scientist
# Mobile-friendly responsive layout included

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
from sklearn import datasets
from io import BytesIO
from PIL import Image

# ---------- Page config ----------
st.set_page_config(page_title="Mohd Azam — Data Scientist", layout="wide", initial_sidebar_state="expanded")

# ---------- Custom CSS (modern + responsive) ----------
st.markdown(
    """
    <style>
    /* page background */
    .stApp { background: linear-gradient(180deg, #0f172a 0%, #071129 100%); color: #e6eef8; }
    /* card */
    .card { background: rgba(255,255,255,0.04); padding: 18px; border-radius: 12px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); }
    .project-title { font-weight:700; font-size:18px; color:#e6eef8 }
    .muted { color: #9fb0d6 }
    .hero { padding:30px; }
    .skill-pill { display:inline-block; margin:4px 6px; padding:6px 10px; background:rgba(255,255,255,0.03); border-radius:999px; border:1px solid rgba(255,255,255,0.05); }
    /* minimal input styling */
    input, textarea { background: rgba(255,255,255,0.02) !important; color: #e6eef8 !important; }
    /* profile image circular + shadow */
    img { border-radius: 50% !important; box-shadow: 0 4px 15px rgba(0,0,0,0.4); }
    /* Responsive text adjustments for small screens */
    @media (max-width: 768px) {
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.4rem !important; }
        .hero { padding: 15px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar / Navigation ----------
st.sidebar.markdown("# Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Skills", "Projects", "Resume", "Contact"])

# ---------- Common helpers ----------
PROFILE_PIC = Image.open("my_image.jpg")  # replace with your actual image file name
NAME = "Mohd Azam"
TITLE = "Data Scientist — Machine Learning | Analytics | MLOps"
LOCATION = "India"

# ---------- Pages ----------
if page == "Home":
    # Use a responsive single column on mobile, two columns on larger screens
    if st.session_state.get("_is_mobile", False):
        col1 = st.container()
        col2 = st.container()
    else:
        col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"<div class='hero'>", unsafe_allow_html=True)
        st.title(f"Hi — I'm {NAME} 👋")
        st.subheader(TITLE)
        st.markdown("""
        I build data products and models that turn raw data into actionable insights. I enjoy
        working on end-to-end ML pipelines, exploratory data analysis, visual storytelling,
        and deploying models that solve real-world problems.
        """)
        st.markdown("**Areas:** Machine Learning, Predictive Modeling, Data Visualization, Feature Engineering, MLOps")
        st.markdown("\n")
        # CTA buttons
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.download_button("Download Resume (PDF)", data=b"", file_name="My_resume.pdf", mime="application/pdf")
        with c2:
            st.button("View Projects")
        with c3:
            st.markdown("[Contact me](#contact)")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card' style='text-align:center'>", unsafe_allow_html=True)
        if PROFILE_PIC:
            st.image(PROFILE_PIC, width=220, caption=NAME, use_column_width=True)
        else:
            st.markdown(f"<h3 style='color:#e6eef8'>{NAME}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p class='muted'>{TITLE}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "About":
    st.header("About Me")
    st.write(
        "I am a data scientist with experience in building machine learning models, data visualizations, and production-ready data pipelines. I love turning ambiguous problems into clean, measurable solutions."
    )
    st.subheader("Experience Snapshot")
    st.markdown("- End-to-end ML model development and deployment\n- Feature engineering and model explainability\n- Dashboarding and storytelling with data")

elif page == "Skills":
    st.header("Skills & Tools")
    st.markdown("""
    <div class='card'>
    <strong>Languages & Libraries</strong>
    <div style='margin-top:8px'>
    <span class='skill-pill'>Python</span>
    <span class='skill-pill'>Pandas</span>
    <span class='skill-pill'>NumPy</span>
    <span class='skill-pill'>scikit-learn</span>
    <span class='skill-pill'>TensorFlow / PyTorch</span>
    <span class='skill-pill'>Altair / Plotly</span>
    </div>
    <br>
    <strong>Tools & Platforms</strong>
    <div style='margin-top:8px'>
    <span class='skill-pill'>Docker</span>
    <span class='skill-pill'>AWS / GCP</span>
    <span class='skill-pill'>Streamlit / Flask</span>
    <span class='skill-pill'>SQL</span>
    <span class='skill-pill'>Airflow</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Projects":
    st.header("Selected Projects")
    # keep charts responsive with use_container_width=True

    # Project 1: Classification demo using Iris dataset
    st.subheader("1) Iris Classification — Model Exploration")
    st.markdown("A quick interactive exploration and simple classifier demo built for demonstration.")

    iris = datasets.load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris['target'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    col1, col2 = st.columns([1,1])
    with col1:
        species = st.selectbox("Color by species", options=list(df_iris['target'].cat.categories))
        filt = df_iris[df_iris['target'] == species]
        st.write(filt.head())
    with col2:
        chart = alt.Chart(df_iris).mark_circle(size=80).encode(
            x=alt.X(iris.feature_names[0], title=iris.feature_names[0]),
            y=alt.Y(iris.feature_names[1], title=iris.feature_names[1]),
            color='target'
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    # Project 2: Time series forecasting demo (synthetic)
    st.subheader("2) Sales Forecasting — Example")
    rng = pd.date_range('2023-01-01', periods=200)
    sales = (np.sin(np.linspace(0, 12, 200)) * 100 + np.linspace(200, 500, 200) + np.random.normal(0, 20, 200)).round(0)
    df_sales = pd.DataFrame({'date': rng, 'sales': sales})
    df_sales['rolling_14'] = df_sales['sales'].rolling(14).mean()

    st.line_chart(df_sales.set_index('date')[['sales','rolling_14']])

    st.markdown("---")
    # Project 3: Interactive model metrics table
    st.subheader("3) Model Comparison")
    metrics = pd.DataFrame({
        'model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'accuracy': [0.86, 0.90, 0.92],
        'f1': [0.84, 0.89, 0.91],
        'inference_ms': [12, 38, 22]
    })
    st.dataframe(metrics)
    fig = px.bar(metrics, x='model', y=['accuracy','f1'], barmode='group', title='Model metrics')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Resume":
    st.header("Resume & More")
    st.write("Download a PDF resume or view a one-page summary below.")
    st.markdown("**Mohd Azam** — Data Scientist")
    st.markdown("- Experience: ML models, data pipelines, dashboards\n- Education: BCA / Relevant degree (edit)\n- Contact: replace with your email")
    resume_text = "Mohd Azam\nData Scientist\nEmail: your@email.com\n\nExperience:\n- ..."
    st.download_button("Download Resume (TXT)", data=resume_text, file_name="resume.txt", mime="text/plain")

elif page == "Contact":
    st.header("Contact")
    st.markdown("If you'd like to work together or have questions, reach out — or send a message below.")
    with st.form("contact_form"):
        name = st.text_input("Your name")
        email = st.text_input("Your email")
        message = st.text_area("Message")
        submitted = st.form_submit_button("Send message")
        if submitted:
            st.success("Thanks for your message — I'll get back to you soon!")

# ---------- Footer ----------
st.markdown("---")
st.markdown("Built with Streamlit. Replace content and projects with your real work.\n\n© 2025 Mohd Azam")

# ---------- How to run (printed when executed as script) ----------
if __name__ == '__main__':
    st.write('This file is a Streamlit app. Run with:')
    st.code('streamlit run data_scientist_portfolio_streamlit.py')