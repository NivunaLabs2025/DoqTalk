import torch
import streamlit as st
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------- Streamlit Config -----------------
st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")
st.title("💬 RAG Chatbot (Llama 3B)")

APP_DIR = Path(__file__).resolve().parent
FAISS_DIR = APP_DIR / "faiss_index"
INDEX_NAME = "index"

# ----------------- Model + Embeddings -----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Llama-3.2-3b-instruct"   # ✅ local model

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})

@st.cache_resource
def load_faiss(_emb):
    return FAISS.load_local(
        folder_path=str(FAISS_DIR),
        embeddings=_emb,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )

@st.cache_resource
def load_llm():
    # Explicitly set the device to "cpu" since no GPU is available
    device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    device_map= device,
    torch_dtype="auto"
    )
    # Ensure the model is explicitly on the CPU
    model.to(device)

    # Return the text generation pipeline
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=False
    )

embeddings = load_embeddings()
db = load_faiss(embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})
llm = load_llm()

# ----------------- Chat History -----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------- Input -----------------
query = st.text_input("Ask me something:")

if query:
    # 1️⃣ Retrieve context from FAISS
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs) if docs else ""

    # 2️⃣ Create Prompt
    prompt = f"""
You are a helpful assistant for a RAG chatbot. 
Answer the following question **only** using the given context. 
If the context does not have the answer, reply exactly:
"The context does not provide this information."
Context:
{context}
Question: {query}
Final Answer:
"""

    # 3️⃣ Run model locally
    response = llm(prompt)
    raw_answer = response[0]["generated_text"]

    # 4️⃣ Extract final answer
    answer = raw_answer.split("Final Answer:", 1)[-1].strip().split("\n")[0]

    # 5️⃣ Save in history
    st.session_state.chat_history.append({"question": query, "answer": answer})

# ----------------- Display History -----------------
if st.session_state.chat_history:
    st.subheader("📜 Conversation History")
    for i, chat in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {chat['question']}")
        st.markdown(f"**A{i}:** {chat['answer']}")
        st.markdown("---")
