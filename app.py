import streamlit as st
import os
import glob

# --- PARCHE PARA STREAMLIT CLOUD (Soluciona errores de base de datos comunes en la nube) ---
# Esto es técnico, pero necesario para que ChromaDB funcione en servidores gratuitos.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Agente IA Soporte", page_icon="🤖")
st.title("🤖 Agente de Atención al Cliente")

# --- 1. GESTIÓN DE LA API KEY (SEGURA) ---
# En la nube, leeremos la clave desde los secretos de Streamlit
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ No se ha configurado la API Key en los secretos de Streamlit.")
    st.stop()

# --- 2. CARGA DE CONOCIMIENTO (TU "ENTRENAMIENTO") ---
# Usamos caché para no procesar los archivos cada vez que el usuario escribe
@st.cache_resource
def load_knowledge_base(api_key):
    docs_path = "documentos"
    
    if not os.path.exists(docs_path):
        return None

    files = glob.glob(os.path.join(docs_path, "*.pdf")) + glob.glob(os.path.join(docs_path, "*.txt"))
    
    if not files:
        return None

    documents = []
    for file_path in files:
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Error leyendo {file_path}: {e}")
            continue

    # Dividir en trozos
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Crear base de datos vectorial en memoria (rápido y gratis)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_documents(chunks, embeddings)
    
    return vectorstore

# Cargar el bot
vectorstore = load_knowledge_base(api_key)

if not vectorstore:
    st.warning("No se encontraron documentos. Añade PDFs o TXT a la carpeta 'documentos'.")
    st.stop()

# --- 3. LÓGICA DEL CHAT ---
llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo", temperature=0)
chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

# Inicializar historial
if "history" not in st.session_state:
    st.session_state.history = []

# Mostrar chat
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu consulta..."):
    # Guardar pregunta usuario
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando documentos..."):
            # Preparar historial para LangChain (formato tuplas)
            chat_history_formatted = [(st.session_state.history[i]["content"], st.session_state.history[i+1]["content"]) 
                                      for i in range(0, len(st.session_state.history)-1, 2)]
            
            result = chain.invoke({"question": prompt, "chat_history": chat_history_formatted})
            response = result['answer']
            st.markdown(response)
            
    st.session_state.history.append({"role": "assistant", "content": response})