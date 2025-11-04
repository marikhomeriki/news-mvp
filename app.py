"""
News Summarizer + Q&A — Streamlit App

Paste a news URL → extract article text → summarize → ask questions.
"""

import os
import re
import numpy as np
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from openai import OpenAI
import logging
logging.basicConfig(
    filename="app.log",        # name of the log file in your project folder
    level=logging.INFO,        # log info and above (warning, error)
    format="%(asctime)s - %(levelname)s - %(message)s"  # optional: timestamp format
)


# -------------- CONFIG --------------
MODEL_SUMMARY = os.getenv("MODEL_SUMMARY", "gpt-4o-mini")
MODEL_QA = os.getenv("MODEL_QA", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# -------------- HELPERS --------------
def fetch_and_extract(url: str):
    """Fetch and extract main article text with fallback, cleanup, paywall detection, and optional JS render."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        logging.warning(f"Failed to fetch HTML from: {url}")
        return "", {"title": "Fetch failed", "url": url}

    # Known paywalls (skip early—don’t hammer)
    paywall_domains = ["nytimes.com", "wsj.com", "bloomberg.com", "thetimes.co.uk"]
    if any(d in url for d in paywall_domains):
        logging.warning(f"Known paywalled domain detected: {url}")
        return "", {"title": "Paywalled article", "url": url}

    # Trafilatura
    result = trafilatura.extract(
        downloaded, include_images=False, include_links=False, favor_recall=True
    )

    if not result:
        # BeautifulSoup fallback
        try:
            from urllib.request import Request, urlopen
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            html = urlopen(req, timeout=15).read().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
            logging.info(f"Used BeautifulSoup fallback for {url}")
        except Exception as e:
            logging.error(f"BeautifulSoup fallback failed for {url}: {e}")
            text = ""
    else:
        text = result or ""

    # If still too short, **optional** JS-rendered fallback (slow)
    if len(text) < 600:
        try:
            from requests_html import HTMLSession
            s = HTMLSession()
            r = s.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.html.render(timeout=20, sleep=1)  # launches headless Chromium; slow
            # collect visible paragraph text
            text = " ".join([e.text for e in r.html.find("p") if e.text])
            logging.info(f"Rendered extraction attempt for {url} (len={len(text)})")
        except Exception as e:
            logging.warning(f"Rendered extraction failed for {url}: {e}")

    # Cleanup
    text = re.sub(r"\s+", " ", text or "").strip()
    seen, uniq = set(), []
    for sent in text.split(". "):
        if sent and sent not in seen:
            seen.add(sent)
            uniq.append(sent)
    text = ". ".join(uniq)

    # Final checks
    if len(text) < 600:
        logging.warning(f"No visible text extracted — likely paywalled or JS-rendered: {url}")
        return "", {"title": "Paywalled or restricted", "url": url}

    meta = trafilatura.extract_metadata(downloaded)
    title = meta.title if meta and getattr(meta, "title", None) else None
    logging.info(f"Fetched article from {url}, length={len(text)}")
    return text, {"title": title, "url": url}






def chunk_text(text: str, max_chars: int = 2800):
    """Split long text into chunks for embeddings."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        if cur_len + len(s) > max_chars and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [s], len(s)
        else:
            cur.append(s)
            cur_len += len(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ Missing OPENAI_API_KEY environment variable.")
        st.stop()
    return OpenAI(api_key=api_key)


def embed_texts(texts):
    client = get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def cosine_sim(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def call_chat_model(system_prompt, user_prompt, model):
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# -------------- STREAMLIT UI --------------
st.set_page_config(page_title="News Summarizer + Q&A", page_icon="📰", layout="wide")
st.title("📰 News Summarizer + Q&A")
st.caption("Paste a news URL → extract → summarize → ask questions.")

url = st.text_input("Paste article URL", placeholder="https://example.com/news/...")
if not url:
    st.stop()

# --- Session state setup (add near top of file, right after imports if you want) ---
if "article_text" not in st.session_state:
    st.session_state.article_text = ""
if "meta" not in st.session_state:
    st.session_state.meta = {}

# --- Fetch article button ---
if st.button("Fetch article"):
    # --- Optional deny list for paywalled sites ---
    deny = ["nytimes.com", "wsj.com", "bloomberg.com", "ft.com", "thetimes.co.uk"]
    if any(d in url for d in deny):
        st.warning("⚠️ This site is usually paywalled. Try another source (BBC, Reuters, Guardian).")
        st.stop()

    with st.spinner("Fetching and extracting article..."):
        text, meta = fetch_and_extract(url)

    # --- Handle paywalled / failed / restricted articles ---
    if not text:
        title = (meta or {}).get("title", "")
        if "Paywalled" in title or "restricted" in title or "Fetch failed" in title:
            st.warning("⚠️ This article appears paywalled or dynamically rendered (login/JS). Try a different source.")
        else:
            st.error("❌ Could not extract article text. Try another URL.")
        st.stop()

    # --- Success path ---
    st.session_state.article_text = text
    st.session_state.meta = meta
    st.success("✅ Article extracted successfully!")



# --- Show article if we have one ---
if st.session_state.article_text:
    meta = st.session_state.meta
    st.write(f"**Title:** {meta.get('title') or 'Unknown'}")
    st.write(f"**Source:** {meta.get('url')}")
    with st.expander("Show extracted text"):
        st.text_area("Extracted Article Text", st.session_state.article_text, height=300)

    # --- Summarize button ---
    if st.button("Summarize Article"):
        with st.spinner("Summarizing..."):
            system_prompt = (
                "You are a precise news analyst. Write a crisp, neutral summary with bullet points, "
                "including: What happened, who is involved, when/where, why it matters, and any numbers. "
                "Avoid hype; be factual."
            )
            user_prompt = (
                f"Summarize the following article in 120–160 words, then add 3 bullet-point key takeaways.\n\n"
                f"{st.session_state.article_text}"
            )
            summary = call_chat_model(system_prompt, user_prompt, MODEL_SUMMARY)
        st.subheader("Summary")
        st.write(summary)


    # ---------- Q&A ----------
st.markdown("---")
st.subheader("Ask questions about this article")

if st.session_state.article_text:
    chunks = chunk_text(st.session_state.article_text)
    chunk_vecs = embed_texts(chunks)

    question = st.text_input("Your question", placeholder="e.g., What are the main policy changes?")
    top_k = st.slider("Number of context chunks", 1, 5, 3)

    if st.button("Answer"):
        with st.spinner("Thinking..."):
            q_vec = embed_texts([question])
            sims = cosine_sim(q_vec, chunk_vecs)[0]
            top_idx = sims.argsort()[::-1][:top_k]
            context = "\n\n".join(f"[Chunk {i+1}]\n{chunks[i]}" for i in top_idx)

            qa_sys = (
                "You are a helpful assistant answering questions strictly using the provided CONTEXT. "
                "If the answer is not in the context, say you don't know. Cite chunk numbers inline like [Chunk 2]."
            )
            qa_user = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer concisely (2–5 sentences)."
            answer = call_chat_model(qa_sys, qa_user, MODEL_QA)

        st.subheader("Answer")
        st.write(answer)
else:
    st.warning("Please fetch an article first.")
    st.stop()

st.markdown("---")
st.caption("Tip: If extraction looks messy, try another site or paste the text directly.")
