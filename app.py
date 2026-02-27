import streamlit as st
import os
import glob

# --- PARCHE PARA STREAMLIT CLOUD (Sqlite) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# --- IMPORTACIONES (Ajustadas a las versiones del requirements) ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Agente IA Soporte", page_icon="🤖")
st.title("🤖 Agente de Atención al Cliente")

# --- 1. GESTIÓN DE LA API KEY (SEGURA) ---
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ No se ha configurado la API Key en los secretos de Streamlit.")
    st.stop()

# --- 2. CARGA DE CONOCIMIENTO ---
@st.cache_resource
def load_knowledge_base(api_key):
    docs_path = "documentos"
    
    if not os.path.exists(docs_path):
        return None

    # Buscar archivos (insensible a mayúsculas/minúsculas)
    all_files = os.listdir(docs_path)
    files = [os.path.join(docs_path, f) for f in all_files if f.lower().endswith(('.pdf', '.txt'))]
    
    if not files:
        return None

    documents = []
    for file_path in files:
        try:
            if file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.lower().endswith(".txt"):
                # Este cambio permite leer archivos con acentos sin error
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
        except Exception as e:
            # Si falla un archivo, lo ignoramos y seguimos
            print(f"Error en {file_path}: {e}")
            continue

    if not documents:
        return None

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_documents(chunks, embeddings)
    
    return vectorstore# Cargar el bot
with st.spinner("Cargando base de conocimiento..."):
    vectorstore = load_knowledge_base(api_key)

if not vectorstore:
    st.warning("👋 No se encontraron documentos. Añade archivos PDF o TXT a la carpeta 'documentos'.")
    st.stop()

# --- 3. LÓGICA DEL CHAT ---
llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo", temperature=0)
chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu consulta..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            chat_history_formatted = []
            for i in range(0, len(st.session_state.history)-1, 2):
                 if i+1 < len(st.session_state.history):
                     q = st.session_state.history[i]["content"]
                     a = st.session_state.history[i+1]["content"]
                     chat_history_formatted.append((q, a))
            
            result = chain.invoke({"question": prompt, "chat_history": chat_history_formatted})
            response = result['answer']
            st.markdown(response)
            
    st.session_state.history.append({"role": "assistant", "content": response})

