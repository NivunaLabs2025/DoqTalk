import os
import re
import time
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Utils: PDF -> text -> chunks
# -----------------------------
def load_pdf_text(uploaded_file) -> str:
    """Read all pages from an uploaded PDF and return plain text."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    text = text.strip()
    if not text:
        raise ValueError("No text found in PDF.")
    return text

_SENT_SPLIT = re.compile(r'(?<=[\.!\?])\s+')

def chunk_text(text: str, max_tokens: int = 200):
    """
    Very simple chunker by sentence count approximation (word-based).
    max_tokens is used as max words per chunk to keep it model-friendly.
    """
    sentences = _SENT_SPLIT.split(text)
    chunks, cur, cur_len = [], [], 0
    for sent in sentences:
        wc = len(sent.split())
        if wc == 0:
            continue
        if cur_len + wc > max_tokens and cur:
            chunks.append(" ".join(cur).strip())
            cur, cur_len = [sent], wc
        else:
            cur.append(sent)
            cur_len += wc
    if cur:
        chunks.append(" ".join(cur).strip())
    # Deduplicate tiny / blanky chunks
    chunks = [c for c in chunks if len(c.split()) >= 5]
    if not chunks:
        raise ValueError("PDF produced no usable chunks.")
    return chunks

# -----------------------------
# Vector + BM25 indexes
# -----------------------------
class FusionStore:
    """
    Holds a FAISS index (cosine similarity using inner product on normalized vectors)
    and a BM25 index over the same chunks. Provides Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2"):
        self.embed_model_name = embed_model_name
        self.embed = SentenceTransformer(self.embed_model_name)
        self.index = None
        self.dim = None
        self.chunk_texts = []
        self.bm25 = None

    def build(self, chunks):
        self.chunk_texts = list(chunks)
        # Build BM25
        tokenized_corpus = [c.split() for c in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Build FAISS (cosine using L2-normalized vectors + IndexFlatIP)
        vecs = self.embed.encode(self.chunk_texts, normalize_embeddings=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype="float32")
        self.dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)

    def search(self, query: str, k: int = 5, rrf_k: int = 60):
        """
        Do BM25 + FAISS retrieval and fuse with RRF.
        Returns top-k chunk strings.
        """
        if self.index is None or self.bm25 is None:
            raise RuntimeError("Index not built.")

        # Vector search
        qv = self.embed.encode([query], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        sims, idxs = self.index.search(qv, min(k * 5, len(self.chunk_texts)))  # grab a few more for fusion
        vec_order = idxs[0].tolist()

        # BM25 search
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_order = np.argsort(-bm25_scores).tolist()

        # Reciprocal Rank Fusion
        ranks = {}
        for rank, ix in enumerate(vec_order):
            ranks[ix] = ranks.get(ix, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, ix in enumerate(bm25_order):
            ranks[ix] = ranks.get(ix, 0.0) + 1.0 / (rrf_k + rank + 1)

        fused = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        top_indices = [ix for ix, _ in fused[:k]]
        return [self.chunk_texts[i] for i in top_indices]

# -----------------------------
# LLM loader (Llama 3.2 3B Instruct)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_llm(model_id: str = "meta-llama/Llama-3.2-3b-instruct"):
    """
    Loads a gated HF model using token from Space secrets (HF_TOKEN).
    Uses chat template for proper formatting.
    """
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        raise ValueError("HF_TOKEN is not set. Add it under Spaces -> Settings -> Secrets.")

    # Authenticate to access gated models
    login(hf_token)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)
    # Ensure pad token id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    return tokenizer, llm

def generate_answer(tokenizer, llm, context: str, question: str, max_new_tokens: int = 300):
    """
    Use the model's chat template for instruction-following.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful tutor. Answer strictly using the provided context. "
                "If the context is insufficient, reply exactly: 'I could not find this in the provided document.'"
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer in 3-7 concise sentences.",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(llm.device)
    # Safety: cap time on CPU
    start = time.time()
    out = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = tokenizer.decode(out[0], skip_special_tokens=True)
    # Heuristic to strip the prompt if model echoes it
    if gen.startswith(prompt):
        gen = gen[len(prompt):].strip()
    # Final tidy
    return gen.strip()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Student Assisted Chatbot (Fusion RAG)",
    page_icon="📚",
    layout="centered",
)
st.title("📚 Student Assisted Chatbot — Fusion RAG (BM25 + FAISS)")

with st.expander("How it works"):
    st.markdown(
        "- Upload a PDF textbook/notes\n"
        "- We chunk the PDF, index with **BM25** and **FAISS**\n"
        "- We fuse both rankings using **RRF** and give the top chunks to **Llama-3.2-3B-Instruct**"
    )

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
question = st.text_input("Your question")

# Cache the store per uploaded file content (hash)
@st.cache_resource(show_spinner=True)
def build_store_from_bytes(pdf_bytes: bytes):
    text = load_pdf_text(uploaded_file=type("F", (), {"read": lambda self=None: pdf_bytes})())
    chunks = chunk_text(text, max_tokens=200)
    store = FusionStore()
    store.build(chunks)
    return store, chunks

if uploaded_file and question:
    try:
        pdf_bytes = uploaded_file.read()
        store, chunks = build_store_from_bytes(pdf_bytes)

        with st.spinner("Retrieving relevant passages..."):
            top_chunks = store.search(question, k=5)
        context = "\n\n".join(f"- {c}" for c in top_chunks)

        tokenizer, llm = load_llm()

        with st.spinner("Thinking with the document..."):
            answer = generate_answer(tokenizer, llm, context, question)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Show retrieved context"):
            for i, c in enumerate(top_chunks, 1):
                st.markdown(f"**Chunk {i}**\n\n{c}")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a PDF and enter a question to begin.")
