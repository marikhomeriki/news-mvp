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
from openai import OpenAI

# -------------- CONFIG --------------
MODEL_SUMMARY = os.getenv("MODEL_SUMMARY", "gpt-4o-mini")
MODEL_QA = os.getenv("MODEL_QA", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# -------------- HELPERS --------------
def fetch_and_extract(url: str):
    """Fetch and extract main article text."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return "", {"title": None, "url": url}

    result = trafilatura.extract(
    downloaded,
    include_images=False,
    include_links=False,
    favor_recall=True,
)

    if not result:
        # Fallback to BeautifulSoup
        try:
            from urllib.request import Request, urlopen
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            html = urlopen(req, timeout=15).read().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = "\n".join(p.get_text(strip=True) for p in soup.find_all("p"))
        except Exception:
            text = ""
    else:
        text = result

    # Title
    meta = trafilatura.extract_metadata(downloaded)
    title = meta.title if meta and getattr(meta, "title", None) else None
    return text or "", {"title": title, "url": url}


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

if st.button("Fetch article"):
    with st.spinner("Fetching and extracting article..."):
        article_text, meta = fetch_and_extract(url)

    if not article_text:
        st.error("Could not extract article text. Try another URL.")
        st.stop()

    st.success("✅ Article extracted successfully!")
    if meta.get("title"):
        st.write(f"**Title:** {meta['title']}")
    st.write(f"**Source:** {meta['url']}")
    with st.expander("Show extracted text"):
        st.text_area("Extracted Article Text", article_text, height=300)

    # ---------- SUMMARY ----------
    if st.button("Summarize Article"):
        with st.spinner("Summarizing..."):
            sys_prompt = (
                "You are a precise news analyst. Write a crisp, neutral summary with bullet points, "
                "including: What happened, who is involved, when/where, why it matters, and any numbers. "
                "Avoid hype; be factual."
            )
            user_prompt = f"Summarize the following article in 120-160 words, then add 3 bullet-point key takeaways.\n\n{article_text}"
            summary = call_chat_model(sys_prompt, user_prompt, MODEL_SUMMARY)
        st.subheader("Summary")
        st.write(summary)

    # ---------- Q&A ----------
    st.markdown("---")
    st.subheader("Ask questions about this article")

    chunks = chunk_text(article_text)
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

st.markdown("---")
st.caption("Tip: If extraction looks messy, try another site or paste the text directly.")
