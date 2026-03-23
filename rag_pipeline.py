import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ── Prompt Template ───────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are DocMind, an intelligent document assistant.
Use the following context from the uploaded document to answer the question.
If the answer is not in the context, say "I couldn't find that in the document."
Always be concise, accurate, and helpful.

Context:
{context}

Question: {question}

Answer:"""
)


class RAGPipeline:
    def __init__(self, pdf_file, llm_provider: str, api_key: str):
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.vectorstore = None
        self.qa_chain = None

        docs = self._load_pdf(pdf_file)
        chunks = self._split_documents(docs)
        self.vectorstore = self._create_vectorstore(chunks)
        self.qa_chain = self._build_chain()

    # ── Load PDF ──────────────────────────────────────────────────────────────
    def _load_pdf(self, pdf_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(pdf_file.read())
            tmp_path = f.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.unlink(tmp_path)
        return docs

    # ── Chunk Documents ───────────────────────────────────────────────────────
    def _split_documents(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " "]
        )
        return splitter.split_documents(docs)

    # ── Embeddings + FAISS Vectorstore ────────────────────────────────────────
    def _create_vectorstore(self, chunks):
        embeddings = self._get_embeddings()
        return FAISS.from_documents(chunks, embeddings)

    def _get_embeddings(self):
        if "OpenAI" in self.llm_provider:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(openai_api_key=self.api_key)

        elif "Gemini" in self.llm_provider:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )

        elif "Groq" in self.llm_provider:
            # Groq doesn't provide embeddings — use HuggingFace (free, local)
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

    # ── LLM Selection ─────────────────────────────────────────────────────────
    def _get_llm(self):
        if "OpenAI" in self.llm_provider:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                openai_api_key=self.api_key
            )

        elif "Gemini" in self.llm_provider:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.2,
                google_api_key=self.api_key
            )

        elif "Groq" in self.llm_provider:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model="llama3-8b-8192",
                temperature=0.2,
                groq_api_key=self.api_key
            )

    # ── Build RetrievalQA Chain ───────────────────────────────────────────────
    def _build_chain(self):
        llm = self._get_llm()
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": RAG_PROMPT},
            return_source_documents=False
        )

    # ── Query ─────────────────────────────────────────────────────────────────
    def query(self, question: str) -> str:
        result = self.qa_chain.invoke({"query": question})
        return result["result"]
