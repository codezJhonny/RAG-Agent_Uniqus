import os, sys, re, json, time, requests
import pdfplumber
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Configuration with SEC website 
HEADERS = {
    "User-Agent": "Vishwanadha Bhanuprakash (bhanu7795@gmail.com)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}
CIKS = {"MSFT": "0000789019", "GOOGL": "0001652044", "NVDA": "0001045810"}
YEARS = ["2022", "2023", "2024"]
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
EMB_CACHE = "embeddings.pkl"

# 1. SEC API
def get_submissions(cik: str):
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/index.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def find_10k(subs, year: str):
    filings = subs.get("filings", {}).get("recent", {})
    for form, date, acc, doc in zip(
        filings.get("form", []),
        filings.get("filingDate", []),
        filings.get("accessionNumber", []),
        filings.get("primaryDocument", [])
    ):
        if form == "10-K" and date.startswith(year):
            return acc.replace("-", ""), doc
    return None, None


def ensure_filings():
    for ticker, cik in CIKS.items():
        try:
            subs = get_submissions(cik)
        except:
            print(f"Could not fetch submissions for {ticker}")
            continue
        for year in YEARS:
            if any(f.startswith(f"{ticker}_{year}") for f in os.listdir(DATA_DIR)):
                continue
            acc, doc = find_10k(subs, year)
            if not acc:
                print(f"No 10-K found for {ticker} {year}")
                continue
            ext = os.path.splitext(doc)[1] or ".htm"
            dest = os.path.join(DATA_DIR, f"{ticker}_{year}{ext}")
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
            print(f"Downloading {ticker} {year} -> {dest}")
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            time.sleep(0.5)


# 2. It will load and process the filings
def load_doc(path):
    if path.endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            return [{"page": i+1, "text": (p.extract_text() or ""), "file": path} for i,p in enumerate(pdf.pages)]
    else:
        html = open(path, "r", encoding="utf-8", errors="ignore").read()
        body = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
        return [{"page": 1, "text": body, "file": path}]


def chunk_text(text, size=300, overlap=100):  # Size is considered 300 for testing but can change to 200 or 150 for faster execution
    words, out, i = text.split(), [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+size]))
        i += size - overlap
    return out


def build_index(files):
    if os.path.exists(EMB_CACHE):
        print("Loading cached embeddings...")
        with open(EMB_CACHE, "rb") as f:
            return pickle.load(f)

    print("Building embeddings... (this may take a couple of minutes)")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    chunks, meta = [], []
    for f in files:
        for p in load_doc(f):
            for ch in chunk_text(p["text"]):
                chunks.append(ch)
                meta.append({"file": f, "page": p["page"], "text": ch})
    embs = model.encode(chunks, normalize_embeddings=True)
    with open(EMB_CACHE, "wb") as f:
        pickle.dump((model, embs, meta), f)
    return model, embs, meta


def search(query, model, embs, meta, topk=5):
    q = model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q, embs)[0]
    idx = sims.argsort()[::-1][:topk]
    return [meta[i] for i in idx]


# 3. Agent
def agent(query, model, embs, meta):
    result = {"query": query, "sub_queries": [query], "reasoning": "", "sources": [], "answer": ""}

    hits = search(query, model, embs, meta)
    if hits:
        result["answer"] = hits[0]["text"]
        result["reasoning"] = "Retrieved most relevant text from filings."
        result["sources"] = [{"file": h["file"], "page": h["page"], "excerpt": h["text"][:200]} for h in hits[:3]]
    else:
        result["answer"] = "No relevant information found."
        result["reasoning"] = "Search returned empty."
    return result


# 4. Main 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<query>\"")
        sys.exit(1)

 
    ensure_filings()    # It will download filings if missing

    # It will load the required files
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith((".pdf",".htm",".html"))]
    if not files:
        print("No filings found even after download.")
        sys.exit(1)

   
    model, embs, meta = build_index(files)  # It will Build or load embeddings

    # It will run the query
    user_query = sys.argv[1]
    res = agent(user_query, model, embs, meta)
    print(json.dumps(res, indent=2))
