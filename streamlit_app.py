import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

TB_DIR = "TextBase"
CV_MODEL_PATH = "architectural_style_classifier.keras"
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = (224, 224)  # must match Notebook 2 exactly

SYSTEM_PROMPT = """You are an expert curator for the National Heritage Preservation Trust, speaking with the authority and warmth of an experienced museum guide.

Rules you must follow:
- Answer using ONLY the context passages provided below. Do not invent facts that are not present in the context.
- If the context does not contain enough information to answer confidently, say so honestly instead of guessing.
- Never open with filler such as "That's a great start!", "Great question!", or similar generic enthusiasm. Begin directly with substantive information.
- Be specific: mention time periods, named examples, and structural or material details from the context wherever relevant, rather than vague generalities.
- If a retrieved passage describes a DIFFERENT architectural style than the one the user is asking about, ignore that passage entirely rather than blending its facts into your answer. Never attribute a feature (e.g. domes, buttresses, mosaics) to a style unless the context explicitly connects that feature to that specific style.
- When you use information from the context, mention which architectural style it relates to, so the user knows where the information comes from.
- If an image classification result is provided in the user's message and its confidence is below 70%, explicitly mention that the prediction is uncertain and could plausibly be a different style.
- Write 3-5 informative sentences: concise, but substantive, like a knowledgeable guide rather than a lecture or a chatbot pleasantry.

Context:
{context}
"""


@st.cache_resource(show_spinner="Setting up knowledge base and models (first run only, may take a minute)...")
def build_pipeline():
    # encoding="utf-8" avoids Windows' default cp1252 encoding choking on special
    # characters (curly quotes, em dashes, etc.) that often appear in pasted text.
    # autodetect_encoding=True is a fallback safety net if a file isn't valid UTF-8.
    loader = DirectoryLoader(
        TB_DIR,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
    )
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # Uses a DIFFERENT persist_directory than Notebook 3's RAG pipeline so the
    # two knowledge bases never merge into one shared, stale vector store.
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db_app")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(model="llama3.2:3b", temperature=0.2)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the conversation history and a follow-up question, "
                    "rewrite the follow-up question as a standalone question "
                    "that includes any necessary context. Do not answer the "
                    "question, only rewrite it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, history_aware_retriever_prompt)
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    cv_model = tf.keras.models.load_model(CV_MODEL_PATH)
    with open(CLASS_INDICES_PATH) as f:
        class_indices = {int(k): v for k, v in json.load(f).items()}

    return rag_chain, cv_model, class_indices


def predict_style(image: Image.Image, cv_model, class_indices):
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img), axis=0).astype("float32")
    preprocessed = preprocess_input(img_array)
    preds = cv_model.predict(preprocessed, verbose=0)[0]
    idx = int(np.argmax(preds))
    return class_indices[idx], float(preds[idx])


# ---- Page setup ----
st.set_page_config(page_title="Heritage Architecture Assistant", page_icon=":classical_building:", layout="wide")

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --cream: #EDE6D6;
    --ink: #2B2A28;
    --stone: #C9BFA8;
    --verdigris: #4C7A6E;
    --terracotta: #A65D3B;
    --gold: #C9A227;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 17px;
}

/* ---- Ambient animated stone/parchment background ---- */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--cream);
    background-image:
        radial-gradient(circle at 20% 20%, rgba(201,191,168,0.55) 0%, transparent 45%),
        radial-gradient(circle at 80% 30%, rgba(76,122,110,0.12) 0%, transparent 50%),
        radial-gradient(circle at 50% 85%, rgba(166,93,59,0.10) 0%, transparent 55%),
        linear-gradient(135deg, #EDE6D6 0%, #E3DAC5 50%, #EDE6D6 100%);
    background-size: 200% 200%, 200% 200%, 200% 200%, 200% 200%;
    animation: driftBackground 32s ease-in-out infinite alternate;
}

@keyframes driftBackground {
    0%   { background-position: 0% 0%, 100% 0%, 50% 100%, 0% 0%; }
    100% { background-position: 15% 10%, 85% 15%, 55% 90%, 20% 20%; }
}

@media (prefers-reduced-motion: reduce) {
    .stApp, [data-testid="stAppViewContainer"] { animation: none; }
}

header[data-testid="stHeader"] { background: transparent; }

/* ---- Hero header ---- */
.hero { text-align: center; padding: 1.2rem 0 0.6rem 0; }
.hero-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--verdigris);
    font-weight: 600;
    margin-bottom: 0.35rem;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 700;
    font-size: 3.6rem;
    color: var(--ink);
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    color: #57534A;
    max-width: 680px;
    margin: 0.7rem auto 0 auto;
    line-height: 1.55;
}
.hero-rule {
    width: 90px;
    height: 2px;
    background: var(--gold);
    margin: 1rem auto 0.4rem auto;
    border-radius: 2px;
}

/* ---- Panel labels ---- */
.panel-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--verdigris);
    font-weight: 700;
    border-bottom: 1px solid rgba(76,122,110,0.35);
    padding-bottom: 0.4rem;
    margin-bottom: 0.9rem;
}

/* ---- Museum-plaque prediction card ---- */
.plaque {
    background: var(--ink);
    border: 1px solid var(--gold);
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    margin-top: 0.8rem;
    text-align: center;
    box-shadow: 0 8px 20px rgba(43,42,40,0.18);
    animation: plaqueFadeIn 0.6s ease-out;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.plaque:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 26px rgba(43,42,40,0.24);
}
@keyframes plaqueFadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
.plaque-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.3rem;
}
.plaque-style {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 600;
    font-size: 1.9rem;
    color: var(--cream);
}
.confidence-bar {
    height: 7px;
    background: rgba(255,255,255,0.15);
    border-radius: 3px;
    margin: 0.65rem 0 0.4rem 0;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    background: var(--gold);
    border-radius: 3px;
    transition: width 0.6s ease;
}
.plaque-confidence {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #C9BFA8;
    margin-top: 0.15rem;
}

/* ---- File uploader ---- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.4);
    border: 1.5px dashed var(--stone);
    border-radius: 8px;
}
[data-testid="stFileUploaderFile"] {
    background: rgba(255,255,255,0.6);
    border-radius: 8px;
}

/* ---- Uploaded image ---- */
[data-testid="stImage"] img {
    border-radius: 8px;
    box-shadow: 0 8px 20px rgba(43,42,40,0.15);
}

/* ---- Captions (e.g. Sources line) ---- */
[data-testid="stCaptionContainer"] {
    color: var(--verdigris) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.01em;
}

/* ---- Buttons ---- */
div.stButton > button {
    background: var(--verdigris);
    color: var(--cream);
    border: none;
    border-radius: 999px;
    padding: 0.45rem 1.2rem;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    transition: background 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}
div.stButton > button:hover {
    background: #3C6358;
    color: var(--cream);
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(43,42,40,0.18);
}

/* ---- Chat messages ---- */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(201,191,168,0.6);
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 3px 10px rgba(43,42,40,0.06);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    font-size: 1.08rem;
    line-height: 1.65;
}

/* ---- Chat input (overriding Streamlit's default red focus ring) ---- */
[data-testid="stChatInput"] {
    border-radius: 10px;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--verdigris) !important;
    box-shadow: 0 0 0 1px var(--verdigris) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif;
    font-size: 1.02rem;
}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">National Heritage Preservation Trust</div>
        <h1 class="hero-title">Heritage Architecture Assistant</h1>
        <p class="hero-subtitle">Upload a photograph of a building and converse with an assistant
        grounded in curated heritage archives &mdash; running entirely on your machine, no cloud API used.</p>
        <div class="hero-rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

rag_chain, cv_model, class_indices = build_pipeline()

# ---- Session state setup (persists across Streamlit re-runs) ----
if "chat_history_obj" not in st.session_state:
    st.session_state.chat_history_obj = InMemoryChatMessageHistory()
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # what gets shown on screen
if "last_image_name" not in st.session_state:
    st.session_state.last_image_name = None
if "last_cv_result" not in st.session_state:
    st.session_state.last_cv_result = None


def get_session_history(session_id):
    return st.session_state.chat_history_obj


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ---- Two-column main layout: inspect panel + chat panel ----
left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    st.markdown('<div class="panel-label">Inspect a Building</div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Architecture photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_image is not None:
        st.image(uploaded_image, caption=None, use_container_width=True)

        # Only re-run the CV model if this is a NEW image, not on every rerun
        if st.session_state.last_image_name != uploaded_image.name:
            image = Image.open(uploaded_image)
            style, confidence = predict_style(image, cv_model, class_indices)
            st.session_state.last_image_name = uploaded_image.name
            st.session_state.last_cv_result = (style, confidence)

        style, confidence = st.session_state.last_cv_result
        st.markdown(
            f"""
            <div class="plaque">
                <div class="plaque-label">Predicted Style</div>
                <div class="plaque-style">{style}</div>
                <div class="confidence-bar"><div class="confidence-fill" style="width:{confidence*100:.1f}%"></div></div>
                <div class="plaque-confidence">{confidence*100:.1f}% confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    if st.button("Reset conversation"):
        st.session_state.chat_history_obj = InMemoryChatMessageHistory()
        st.session_state.display_messages = []
        st.session_state.last_image_name = None
        st.session_state.last_cv_result = None
        st.rerun()

with right_col:
    st.markdown('<div class="panel-label">Ask the Assistant</div>', unsafe_allow_html=True)

    for msg in st.session_state.display_messages:
        avatar = "🏛" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ---- Quick-question chips: one click fills in a common question ----
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    suggested_questions = [
        "What time period is this from?",
        "What materials were used?",
        "What preservation challenges does it face?",
    ]
    chip_cols = st.columns(len(suggested_questions))
    for chip_col, question_text in zip(chip_cols, suggested_questions):
        with chip_col:
            if st.button(question_text, key=f"chip_{question_text}", use_container_width=True):
                st.session_state.pending_question = question_text

    typed_question = st.chat_input("Ask about this building or any architectural style...")
    user_question = typed_question or st.session_state.pending_question
    st.session_state.pending_question = None

    if user_question:
        st.session_state.display_messages.append({"role": "user", "content": user_question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_question)

        augmented_input = user_question
        if st.session_state.last_cv_result is not None:
            style, confidence = st.session_state.last_cv_result
            augmented_input = (
                f"[Image analysis result: the uploaded image was classified as "
                f"'{style}' with {confidence*100:.1f}% confidence.] "
                f"User's question: {user_question}"
            )

        with st.chat_message("assistant", avatar="🏛"):
            with st.spinner("Thinking..."):
                response = conversational_rag_chain.invoke(
                    {"input": augmented_input},
                    config={"configurable": {"session_id": "streamlit_session"}},
                )
                answer = response["answer"]
                sources = sorted({doc.metadata.get("source", "unknown") for doc in response["context"]})

                st.markdown(answer)
                if sources:
                    st.caption("Sources: " + ", ".join(Path(s).name for s in sources))

        st.session_state.display_messages.append({"role": "assistant", "content": answer})
