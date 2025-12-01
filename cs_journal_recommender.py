"""
UG/PG CS Core Subjects Journal Recommendation System
- Student types topic → relevant research papers show
- Modern topics (ML, AI, Networks, Security) → arXiv PDFs
- Classic topics (DSA, OS, DB, Compiler) → metadata only using fallback keywords
- TF-IDF ranking on title+abstract
Requirements: requests, pandas, numpy, scikit-learn
Run: python cs_journal_recommender_all.py
"""

import requests
import pandas as pd
import numpy as np
import time
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---------------- CONFIG ----------------
ARXIV_URL = "http://export.arxiv.org/api/query"
MAX_RESULTS = 100  # max papers to fetch per query

# UG + PG Core CS Subjects
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

# ---------------- Fetch arXiv papers ----------------
def fetch_arxiv_papers(topic, category, keywords):
    print(f"Fetching papers for '{topic}' from arXiv...")
    all_entries = []
    search_query = f"cat:{category}+AND+all:({keywords})"
    params = {"search_query": search_query, "start": 0, "max_results": MAX_RESULTS}
    r = requests.get(ARXIV_URL, params=params, timeout=30)
    if r.status_code != 200:
        print(f"Error fetching: {r.status_code}")
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
    df = pd.DataFrame(all_entries)
    if df.empty:
        print("No papers found. Try another topic.")
    else:
        print(f"Fetched {len(df)} papers.")
    return df

# ---------------- TF-IDF & Recommendation ----------------
def build_tfidf(df):
    vect = TfidfVectorizer(max_features=10000, stop_words='english')
    X = vect.fit_transform(df['text'])
    return vect, X

def recommend_papers(query, df, vect, X, top_k=10):
    qv = vect.transform([query])
    sims = linear_kernel(qv, X).flatten()
    idx = np.argsort(-sims)[:top_k]
    res = df.iloc[idx].copy()
    return res.reset_index(drop=True)

# ---------------- Main ----------------
def main():
    print("=== UG/PG CS Core Subjects Journal Recommendation ===")
    print("Available subjects:")
    for sub in CS_SUBJECTS:
        print(f"- {sub}")
    
    subject = input("Enter your subject/topic: ").strip()
    if subject not in CS_SUBJECTS:
        print("Unknown subject. Using general CS with keywords 'computer science'.")
        category = "cs"
        keywords = "computer science"
    else:
        category = CS_SUBJECTS[subject]["category"]
        keywords = CS_SUBJECTS[subject]["keywords"]
    
    df = fetch_arxiv_papers(subject, category, keywords)
    if df.empty:
        return
    
    vect, X = build_tfidf(df)
    
    while True:
        query = input("\nEnter search query (or 'exit' to quit): ").strip()
        if query.lower() in ("exit","quit"):
            print("Exiting. Bye!")
            break
        topk_s = input("Top how many results? (default 10): ").strip()
        try:
            top_k = int(topk_s) if topk_s else 10
        except:
            top_k = 10
        
        recs = recommend_papers(query, df, vect, X, top_k)
        print(f"\nTop {top_k} results for '{query}':\n")
        for i, r in recs.iterrows():
            print("--------------------------------------------------")
            print(f"{i+1}. Title: {r['title']}")
            print(f"   Authors: {r['authors']}")
            print(f"   Year: {r['year']}")
            print(f"   PDF Link: {r['pdf_link'] if r['pdf_link'] else 'Not available'}")
            abstract_snip = (r['abstract'][:300]+"...") if len(r['abstract'])>300 else r['abstract']
            print(f"   Abstract snippet: {abstract_snip}")
        print("--------------------------------------------------")

if __name__ == "__main__":
    main()
