
from pathlib import Path

import streamlit as st

from src.pdf_processor import (
    extract_text_from_pdf,
    chunk_text,
    remove_references
)
from src.rag_pipeline import RAGPipeline


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Paper Assistant")

st.write(
    "Upload, explore, and compare research papers "
    "using Retrieval-Augmented Generation."
)


# --------------------------------------------------
# Initialize RAG pipeline
# --------------------------------------------------

@st.cache_resource
def load_rag():
    return RAGPipeline()


rag = load_rag()


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "ask_messages" not in st.session_state:
    st.session_state.ask_messages = []

if "compare_messages" not in st.session_state:
    st.session_state.compare_messages = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Research Papers")

    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button("Process PDFs", type="primary"):

            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)

            with st.spinner("Processing documents..."):

                for uploaded_file in uploaded_files:

                    # Save PDF locally.
                    pdf_path = (
                        data_dir / Path(uploaded_file.name).name
                    )

                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Extract and process text.
                    pages = extract_text_from_pdf(pdf_path)

                    pages = remove_references(pages)

                    chunks = chunk_text(pages)

                    if chunks:

                        rag.vector_store.add_chunks(chunks)

                        st.success(
                            f"{uploaded_file.name}: "
                            f"{len(chunks)} chunks indexed"
                        )

                    else:

                        st.warning(
                            f"No text extracted from "
                            f"{uploaded_file.name}"
                        )

    st.divider()

    # ----------------------------------------------
    # Document selection
    # ----------------------------------------------

    st.subheader("Search Documents")

    available_documents = (
        rag.vector_store.list_documents()
    )

    selected_documents = st.multiselect(
        "Select papers to search",
        options=available_documents,
        default=available_documents
    )

    st.caption(
        f"{len(available_documents)} papers indexed"
    )

    st.divider()

    # ----------------------------------------------
    # Assistant mode
    # ----------------------------------------------

    st.subheader("Assistant Mode")

    mode = st.radio(
        "Choose how to explore your papers:",
        ["Ask Questions", "Compare Papers"],
        horizontal=False
    )

    if mode == "Compare Papers":

        if len(selected_documents) < 2:
            st.warning(
                "Select at least two papers to compare."
            )

    st.divider()

    # ----------------------------------------------
    # Clear conversation
    # ----------------------------------------------

    if st.button("🗑️ Clear Conversation"):

        if mode == "Ask Questions":
            st.session_state.ask_messages = []

        else:
            st.session_state.compare_messages = []

        st.rerun()


# --------------------------------------------------
# Select active conversation history
# --------------------------------------------------

if mode == "Ask Questions":

    messages = st.session_state.ask_messages

else:

    messages = st.session_state.compare_messages


# --------------------------------------------------
# Main interface
# --------------------------------------------------

if mode == "Ask Questions":

    st.subheader("💬 Ask Questions")

    st.write(
        "Ask questions about the selected research papers. "
        "Answers are grounded in retrieved document excerpts."
    )

else:

    st.subheader("🔍 Compare Papers")

    st.write(
        "Compare the methods, contributions, and findings "
        "of two or more research papers."
    )


# --------------------------------------------------
# Display conversation history
# --------------------------------------------------

for message in messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if "sources" in message:

            with st.expander("📖 View Sources"):

                for i, source in enumerate(
                    message["sources"], start=1
                ):

                    st.markdown(
                        f"**[{i}] {source['source']} "
                        f"— Page {source['page']}**"
                    )

                    st.write(source["text"])

                    st.divider()


# --------------------------------------------------
# Validate selected documents
# --------------------------------------------------

can_submit = True

if not selected_documents:

    st.info(
        "Select at least one document to ask questions."
    )

    can_submit = False

elif (
    mode == "Compare Papers"
    and len(selected_documents) < 2
):

    st.info(
        "Select at least two documents to compare."
    )

    can_submit = False


# --------------------------------------------------
# Chat input
# --------------------------------------------------

if mode == "Ask Questions":

    placeholder = (
        "Ask a question about your research papers..."
    )

else:

    placeholder = (
        "What would you like to compare between these papers?"
    )


question = st.chat_input(
    placeholder,
    disabled=not can_submit
)


# --------------------------------------------------
# Process user question
# --------------------------------------------------

if question and can_submit:

    # Save user message.
    messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching papers and generating answer..."
        ):

            try:

                # ----------------------------------
                # Choose RAG mode
                # ----------------------------------

                if mode == "Compare Papers":

                    result = rag.compare_papers(
                        question=question,
                        sources=selected_documents
                    )

                else:

                    result = rag.answer_question(
                        question=question,
                        top_k=5,
                        sources=selected_documents
                    )

                # ----------------------------------
                # Display generated answer
                # ----------------------------------

                st.markdown(result["answer"])

                # ----------------------------------
                # Display numbered sources
                # ----------------------------------

                if result["sources"]:

                    with st.expander("📖 View Sources"):

                        for i, source in enumerate(
                            result["sources"], start=1
                        ):

                            st.markdown(
                                f"**[{i}] {source['source']} "
                                f"— Page {source['page']}**"
                            )

                            st.write(source["text"])

                            st.divider()

                # ----------------------------------
                # Save assistant response
                # ----------------------------------

                messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })

            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )
