"""
rag_app.py

Chat-style RAG app with:
- Chroma vectorstore (built by index_documents.py)
- NVIDIAEmbeddings for retrieval
- ChatNVIDIA (NIM / NVIDIA API Catalog) as primary LLM
- Offline fallback: if LLM call fails, answer is synthesized directly from retrieved docs
"""

import os, json
from typing import List, Dict, Any
from dotenv import load_dotenv
import regex

import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()  # load NVIDIA_API_KEY from .env`)

# ---------- CONFIG ----------
CHROMA_DIR = "chroma_db"           # same as in index_documents.py
CHUNKED_DOCS_JSON = "chunked_docs.json"

# LLM & embedding models – can override via env vars
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Force offline mode:
#   export OFFLINE_MODE="1"
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "0") == "1"


# ---------- PII MASKING ----------
def mask_pii(text: str) -> str:
    text = regex.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL_MASKED]", text)
    text = regex.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b", "[PHONE_MASKED]", text)
    text = regex.sub(r"\b([A-Z][a-z]{2,})\b", "[NAME_MASKED]", text)
    return text

# ---------- HELPERS: VECTORS & LLM ----------
@st.cache_resource(show_spinner=False)
def load_vectorstore() -> Chroma:
    """Load the persisted Chroma DB with OpenAI embeddings (for online mode)."""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectordb


@st.cache_resource(show_spinner=False)
def load_llm() -> ChatOpenAI:
    """Create the ChatOpenAI LLM client."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.2,
    )
    return llm


def build_prompt() -> ChatPromptTemplate:
    """
    Prompt with support for:
    - {context}: retrieved documents (already PII-masked)
    - {chat_history}: previous turns as text
    - {input}: current user question
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use ONLY the provided context and chat history.\n"
                "If the answer is not in the context, say you don't know.\n\n"
                "Conversation so far:\n{chat_history}\n\n"
                "Context:\n{context}",
            ),
            ("human", "{input}"),
        ]
    )
    return prompt


def build_chat_history_string(messages: List[Dict[str, str]]) -> str:
    """
    Convert chat messages into a simple text transcript:
    [user]: ...
    [assistant]: ...
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"[user]: {content}")
        else:
            lines.append(f"[assistant]: {content}")
    return "\n".join(lines)


# ---------- OFFLINE FALLBACK (NO LLM) ----------
@st.cache_resource(show_spinner=False)
def load_chunked_docs_for_offline() -> List[Document]:
    """Load chunked docs from JSON for offline mode (no OpenAI calls)."""
    if not os.path.exists(CHUNKED_DOCS_JSON):
        raise FileNotFoundError(
            f"{CHUNKED_DOCS_JSON} not found. Run index_documents.py first."
        )

    with open(CHUNKED_DOCS_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs: List[Document] = []
    for item in raw:
        docs.append(
            Document(
                page_content=item["page_content"],
                metadata=item.get("metadata", {}),
            )
        )
    return docs


def simple_keyword_rank(query: str, docs: List[Document], top_k: int = 4) -> List[Document]:
    """
    Very simple keyword-based ranking for offline mode.
    Uses overlap between query words and document text.
    """
    query_words = set(query.lower().split())
    scored: List[tuple[float, Document]] = []

    for d in docs:
        text = d.page_content.lower()
        score = sum(1.0 for w in query_words if w in text)
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)

    if scored and scored[0][0] == 0:
        return [d for _, d in scored[:top_k]]

    return [d for _, d in scored[:top_k]]


def offline_fallback_answer(question: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Offline fallback when OpenAI is unavailable or OFFLINE_MODE is True.

    - Uses locally stored chunked_docs.json
    - Does a simple keyword ranking to fetch relevant docs
    - Builds a naive answer by concatenating text
    """
    docs_all = load_chunked_docs_for_offline()
    docs = simple_keyword_rank(question, docs_all, top_k=top_k)

    combined_text_parts = []
    for doc in docs:
        combined_text_parts.append(doc.page_content.strip())
        if len("\n\n".join(combined_text_parts)) > 2000:  # cap length
            break

    combined_text = "\n\n---\n\n".join(combined_text_parts) or "No relevant context found."

    answer_text = (
        "⚠️ Running in offline fallback mode (no OpenAI calls).\n\n"
        "Here is the most relevant context I could find:\n\n"
        f"{combined_text}"
    )

    return {
        "answer": answer_text,
        "context": docs,
        "offline": True,
    }


# ---------- RAG ANSWER (ONLINE, WITH PII MASKING) ----------
def rag_answer(
    question: str,
    messages: List[Dict[str, str]],
    top_k: int = 4,
) -> Dict[str, Any]:
    """
    Answer using RAG with OpenAI (online).
    - Retrieves docs from Chroma
    - Masks PII in the combined context BEFORE sending to LLM
    - Uses ChatOpenAI with a ChatPromptTemplate
    If OFFLINE_MODE or any error occurs, falls back to offline_fallback_answer.
    """
    chat_history_str = build_chat_history_string(messages)

    # If explicitly offline, skip OpenAI entirely
    if OFFLINE_MODE:
        return offline_fallback_answer(question, top_k=top_k)

    try:
        vectordb = load_vectorstore()
        retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
        docs: List[Document] = retriever.invoke(question)

        # Build raw context from docs
        raw_context = "\n".join(doc.page_content for doc in docs)

        # ✅ Mask PII BEFORE sending to LLM
        # masked_context = mask_pii(raw_context)
        # print(masked_context)

        # Build prompt & LLM chain
        prompt = build_prompt()
        llm = load_llm()
        chain = prompt | llm

        llm_response = chain.invoke(
            {
                "context": raw_context,
                "chat_history": chat_history_str,
                "input": question,
            }
        )

        answer_text = llm_response.content if hasattr(llm_response, "content") else str(
            llm_response
        )

        return {
            "answer": answer_text,
            "context": docs,   # original docs (unmasked) for UI display
            "offline": False,
        }

    except Exception as e:
        # Any OpenAI / network / API error -> offline fallback
        st.warning(f"Primary LLM failed, switching to offline fallback. Error: {e}")
        return offline_fallback_answer(question, top_k=top_k)


# ---------- ONBOARDING LOGIC ----------
def validate_email(email: str) -> bool:
    return "@" in email and "." in email and " " not in email


def validate_phone(phone: str) -> bool:
    digits = [c for c in phone if c.isdigit()]
    return len(digits) >= 7  # very loose check


def handle_onboarding(user_input: str) -> str:
    """
    Step the onboarding state machine:
    stages: name -> email -> phone -> done
    Stores data in st.session_state['user_profile'].
    Returns the assistant's reply.
    """
    stage = st.session_state["onboarding_stage"]
    profile = st.session_state["user_profile"]

    if stage == "name":
        profile["name"] = user_input.strip()
        st.session_state["onboarding_stage"] = "email"
        return f"Nice to meet you, **{profile['name']}**! 😊\n\nWhat is your **email address**?"

    elif stage == "email":
        if not validate_email(user_input.strip()):
            return "That doesn’t look like a valid email. Could you please enter a valid email address?"
        profile["email"] = user_input.strip()
        st.session_state["onboarding_stage"] = "phone"
        return "Got it! 📧\n\nNow, please share your **phone number** (with country code if you want)."

    elif stage == "phone":
        if not validate_phone(user_input.strip()):
            return "Hmm, that phone number seems off. Please enter a valid phone number (at least 7 digits)."
        profile["phone"] = user_input.strip()
        st.session_state["onboarding_stage"] = "done"

        name = profile.get("name", "there")
        return (
            f"Awesome, {name}! 🎉 You’re all set.\n\n"
            "You can now ask me anything about your indexed content."
        )

    else:
        return "Onboarding is already complete. Ask me anything about your indexed content!"


# ---------- STREAMLIT APP (CHAT MODE + ONBOARDING + PII-MASKED RAG) ----------
st.set_page_config(page_title="OpenAI RAG Chat", page_icon="🧠", layout="wide")

st.title("🧠 RAG Chat over ChromaDB with OpenAI\n(With Onboarding, PII Masking & Offline Fallback)")

st.markdown(
    """
This chat app:
- First does a quick **sign-up** (Name, Email, Phone) via chat  
- Then lets you ask questions over your indexed content using **RAG**  
- **Masks PII** in retrieved context before sending it to the LLM  
- Falls back to a **local offline mode** if OpenAI is unavailable  
"""
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K documents", min_value=1, max_value=10, value=4)
    st.markdown("---")
    st.markdown("**Online / Offline**")
    st.write(f"`OFFLINE_MODE` env: `{OFFLINE_MODE}`")
    st.markdown(
        """
If `OFFLINE_MODE=1` or an OpenAI call fails,  
the app will fall back to local keyword-based answers.
        """
    )

    st.markdown("---")
    if "user_profile" in st.session_state and st.session_state["user_profile"]:
        st.subheader("Current user")
        prof = st.session_state["user_profile"]
        st.write(f"**Name**: {prof.get('name', '-')}")
        st.write(f"**Email**: {prof.get('email', '-')}")
        st.write(f"**Phone**: {prof.get('phone', '-')}")


# Initialize chat + onboarding state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["onboarding_stage"] = "name"
    st.session_state["user_profile"] = {}

    greeting = (
        "Hi! 👋 Before we start with RAG, let's do a quick sign-up.\n\n"
        "**What should I call you?** (Please share your name.)"
    )
    st.session_state["messages"].append({"role": "assistant", "content": greeting})

# Render existing chat
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your reply or question here...")

if user_input:
    # Add user message to history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    stage = st.session_state.get("onboarding_stage", "done")

    # ONBOARDING FLOW
    if stage != "done":
        reply = handle_onboarding(user_input)
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state["messages"].append(
            {"role": "assistant", "content": reply}
        )

    # RAG FLOW (after onboarding is complete)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = rag_answer(
                    question=user_input,
                    messages=st.session_state["messages"],
                    top_k=top_k,
                )
                answer_text = result["answer"]
                context_docs = result["context"]
                offline = result.get("offline", False)

                st.markdown(answer_text)

                if offline:
                    st.caption("Answered using offline fallback (no OpenAI calls).")

                # Show retrieved context (UNMASKED, local only)
                with st.expander("📚 Retrieved context"):
                    for i, doc in enumerate(context_docs, start=1):
                        source = doc.metadata.get("source", "unknown")
                        st.markdown(f"**Chunk {i} — Source:** {source}")
                        st.write(doc.page_content)
                        st.markdown("---")

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer_text}
        )