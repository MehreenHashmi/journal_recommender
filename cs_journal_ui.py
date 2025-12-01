import streamlit as st
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---------------- CONFIG ----------------
ARXIV_URL = "http://export.arxiv.org/api/query"
MAX_RESULTS = 50

CS_SUBJECTS = {
    "DSA": {"category": "cs", "keywords": "NP-complete OR graph algorithms OR dynamic programming"},
    "Database Systems": {"category": "cs.DB", "keywords": "database OR SQL OR NoSQL"},
    "Operating Systems": {"category": "cs.OS", "keywords": "operating systems OR scheduling OR deadlock"},
    "Computer Networks": {"category": "cs.NI", "keywords": "computer networks OR congestion OR routing"},
    "Computer Architecture": {"category": "cs.AR", "keywords": "computer architecture OR microprocessor OR cache"},
    "Machine Learning": {"category": "cs.LG", "keywords": "machine learning OR neural networks OR deep learning"},
    "Software Engineering": {"category": "cs.SE", "keywords": "software engineering OR software design OR agile"},
    "Cybersecurity": {"category": "cs.CR", "keywords": "cybersecurity OR cryptography OR malware"},
    "Programming Languages": {"category": "cs.PL", "keywords": "programming languages OR compiler OR interpreter"},
    "Compiler Design": {"category": "cs.PL", "keywords": "compiler design OR parsing OR code generation"},
    "Web Technologies": {"category": "cs.SE", "keywords": "web development OR web technologies OR frontend"},
    "Computer Graphics": {"category": "cs.GR", "keywords": "computer graphics OR rendering OR 3D"},
    "Cloud Computing": {"category": "cs.DC", "keywords": "cloud computing OR distributed systems OR virtualization"},
    "Embedded Systems": {"category": "cs.ET", "keywords": "embedded systems OR IoT OR real-time"}
}

# ---------------- Fetch arXiv ----------------
@st.cache_data
def fetch_arxiv_papers(topic, category, keywords):
    all_entries = []
    search_query = f"cat:{category}+AND+all:({keywords})"
    params = {"search_query": search_query, "start": 0, "max_results": MAX_RESULTS}
    r = requests.get(ARXIV_URL, params=params, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()

    root = ET.fromstring(r.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)

    for e in entries:
        title = e.find("atom:title", ns).text.strip().replace("\n", " ")
        abstract = e.find("atom:summary", ns).text.strip().replace("\n", " ")
        pdf_link = None
        for link in e.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_link = link.attrib["href"]
        authors = ", ".join([a.find("atom:name", ns).text for a in e.findall("atom:author", ns)])
        published = e.find("atom:published", ns).text[:10]
        all_entries.append({
            "title": title,
            "abstract": abstract,
            "text": title + ". " + abstract,
            "authors": authors,
            "year": published[:4],
            "pdf_link": pdf_link
        })
    return pd.DataFrame(all_entries)


# ---------------- TF-IDF ----------------
@st.cache_data
def build_tfidf(df):
    vect = TfidfVectorizer(max_features=10000, stop_words='english')
    X = vect.fit_transform(df['text'])
    return vect, X

def recommend_papers(query, df, vect, X, top_k=10):
    qv = vect.transform([query])
    sims = linear_kernel(qv, X).flatten()
    idx = sims.argsort()[-top_k:][::-1]
    res = df.iloc[idx].copy()
    return res.reset_index(drop=True)


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="CS Journal Recommender", layout="wide")

# Custom CSS — more beautiful
st.markdown("""
<style>
/* MAIN BACKGROUND */
body {
    background-color: #f7f9fc !important;
    font-family: 'Inter', sans-serif;
}

/* NAVBAR */
.navbar {
    background: linear-gradient(90deg, #dbe9ff, #eaf2ff);
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 25px;
    border: 1px solid #c7d9ff;
}
.title-text {
    color: #002855;
    font-size: 32px;
    font-weight: 800;
    letter-spacing: 0.5px;
}

/* PAPER CARD */
.paper-card {
    background: #ffffff;
    padding: 20px 22px;
    border-radius: 14px;
    border: 1px solid #e1e8f0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.04);
    margin-bottom: 20px;
    transition: 0.2s ease-in-out;
}
.paper-card:hover {
    transform: scale(1.01);
    box-shadow: 0 4px 14px rgba(0,0,0,0.07);
}

/* TITLES */
.paper-title {
    font-size: 20px;
    color: #003366;
    font-weight: 700;
}
.paper-meta {
    font-size: 14px;
    color: #555;
}

/* SOFT BLUE BUTTON */
.pdf-btn {
    background-color: #4b8fe2;
    color: white !important;
    padding: 8px 14px;
    border-radius: 6px;
    text-decoration: none;
    font-size: 14px;
}
.pdf-btn:hover {
    background-color: #3c7bcc;
}

/* EXPANDER STYLE */
details summary {
    font-size: 15px;
    font-weight: 600;
    color: #003366;
}

</style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("<div class='navbar'><span class='title-text'>📚 Journal Research Paper Recommendation (CS Core Subjects)</span></div>", unsafe_allow_html=True)

st.write("Enter your query below to discover relevant research papers from arXiv.")

# Subject selection
subject = st.selectbox("Select Subject:", list(CS_SUBJECTS.keys()))

subject_data = CS_SUBJECTS[subject]
df = fetch_arxiv_papers(subject, subject_data["category"], subject_data["keywords"])

if df.empty:
    st.warning("No papers found for this subject. Try another.")
else:
    vect, X = build_tfidf(df)

    query = st.text_input("Enter search query:")
    top_k = st.slider("Number of top results:", 1, 20, 10)

    if st.button("Search") and query.strip() != "":
        recs = recommend_papers(query, df, vect, X, top_k)
        st.write(f"### Showing Top {len(recs)} results")

        for i, r in recs.iterrows():
            st.markdown(
                f"""
                <div class="paper-card">
                    <div class="paper-title">{i+1}. {r['title']}</div>
                    <div class="paper-meta">
                        <b>Authors:</b> {r['authors']} &nbsp; | &nbsp;
                        <b>Year:</b> {r['year']}
                    </div><br>
                """,
                unsafe_allow_html=True
            )

            # Updated lighter, visible blue button
            if r['pdf_link']:
                st.markdown(f"<a class='pdf-btn' href='{r['pdf_link']}' target='_blank'>Open PDF</a>", unsafe_allow_html=True)

            with st.expander("Abstract"):
                st.write(r['abstract'])

            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; margin-top:30px; font-size:15px; color:#444;'>
    <b>Project by:</b> Mehreen Hashmi (01102102025) • Namrata Kaushik (01502102025) • Sonali (02202102025)
</div>
""", unsafe_allow_html=True)
