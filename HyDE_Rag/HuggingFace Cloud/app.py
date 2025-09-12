import os
import gradio as gr
import fitz  # PyMuPDF for PDFs
import docx
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from gtts import gTTS
from huggingface_hub import login

# =============================
# 1) Auth & Config
# =============================
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("⚠️ Please set your HF_TOKEN as an environment variable.")

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_ID = "meta-llama/Llama-3.2-3b-instruct"
ASR_MODEL_ID = "openai/whisper-small"

# =============================
# 2) Load Models
# =============================
embedding_model = SentenceTransformer(EMBED_MODEL_ID)

login(HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, token=HF_TOKEN)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HF_TOKEN
)

# Whisper (speech-to-text)
stt_model = pipeline("automatic-speech-recognition", model=ASR_MODEL_ID, token=HF_TOKEN)

# =============================
# 3) File Text Extraction
# =============================
def extract_text(file_path: str) -> str:
    if not file_path:
        return ""
    _, ext = os.path.splitext(file_path.lower())
    text = ""
    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text("text")
    elif ext == ".docx":
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        with open(file_path, "rb") as f:
            text = f.read().decode("utf-8", errors="ignore")
    return text

# =============================
# 4) Build FAISS Index
# =============================
def build_faiss(text: str, chunk_size=500, overlap=50):
    if not text.strip():
        return None, None

    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    if not chunks:
        return None, None

    embeddings = embedding_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, chunks

# =============================
# 5) Globals (indexed docs)
# =============================
doc_index = None
doc_chunks = None

# =============================
# 6) Handlers
# =============================
def upload_file(file_path: str):
    global doc_index, doc_chunks
    if not file_path:
        return "⚠️ Please upload a file first."
    text = extract_text(file_path)
    idx, chunks = build_faiss(text)
    if idx is None:
        return "⚠️ Could not index: file appears empty."
    doc_index, doc_chunks = idx, chunks
    return f"✅ Document indexed! {len(chunks)} chunks ready."

def answer_query(query: str):
    global doc_index, doc_chunks
    if not query or not query.strip():
        return "⚠️ Please enter a question."
    if doc_index is None or not doc_chunks:
        return "⚠️ Please upload and index a document first."

    # ---- Retrieve context ----
    q_vec = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = doc_index.search(q_vec, k=min(5, len(doc_chunks)))
    retrieved = [doc_chunks[i] for i in I[0] if 0 <= i < len(doc_chunks)]
    context = "\n".join(retrieved)

    # ---- Final Answer ----
    final_prompt = f"""
    [INST] You are a helpful tutor. Based only on the context below, answer the question.
    If not in context, say "I could not find this in the text."
    Context:
    {context}
    Question: {query}
    Answer: [/INST]
    """
    inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True).to(llm.device)
    outputs = llm.generate(**inputs, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return answer

def synthesize_with_gtts(text: str, out_path="out.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(out_path)
    return out_path

def voice_query(audio_path: str):
    if not audio_path:
        return "⚠️ Please record your question.", "", None

    # 1) Speech -> Text
    asr = stt_model(audio_path)
    recognized = asr.get("text", "").strip()
    if not recognized:
        return "⚠️ Could not transcribe audio.", "", None

    # 2) Answer Query
    ans = answer_query(recognized)

    # 3) Text -> Speech
    mp3_path = synthesize_with_gtts(ans, "answer.mp3")

    return recognized, ans, mp3_path

# =============================
# 7) Gradio UI
# =============================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan")) as demo:
    gr.Markdown("# 📚 RAG Chatbot + 🎤 Voice (Whisper + gTTS)")
    gr.Markdown("Upload a PDF/DOCX/TXT and ask by typing **or** speaking.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📂 Upload Document", type="filepath")
            upload_btn = gr.Button("⚡ Index Document", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### ✍️ Text Chat")
            query = gr.Textbox(label="❓ Ask a Question", placeholder="e.g., What are the key points?")
            ask_btn = gr.Button("🚀 Get Answer", variant="primary")
            answer = gr.Textbox(label="💡 Answer", lines=8)

            gr.Markdown("### 🎤 Voice Chat")
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Speak your question")
            rec_text = gr.Textbox(label="📝 Recognized Speech", interactive=False)
            v_answer = gr.Textbox(label="💡 Answer (voice)", lines=8)
            v_audio = gr.Audio(label="🔊 Bot Voice Reply")

    # Bind events
    upload_btn.click(fn=upload_file, inputs=file_input, outputs=status)
    ask_btn.click(fn=answer_query, inputs=query, outputs=answer)
    mic_input.change(fn=voice_query, inputs=mic_input, outputs=[rec_text, v_answer, v_audio])

demo.launch()

