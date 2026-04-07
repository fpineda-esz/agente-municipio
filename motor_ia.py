import os
# --- PARCHE PARA STREAMLIT CLOUD ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -----------------------------------
from dotenv import load_dotenv
from operator import itemgetter

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Cargar variables de entorno
load_dotenv()

def crear_agente():
    try:
        # Forzamos a buscar la carpeta "tramites" en la misma ruta
        ruta_base = os.path.dirname(os.path.abspath(__file__))
        ruta_tramites = os.path.join(ruta_base, 'tramites')
        
        # Prevenir error si la carpeta no existe aún
        if not os.path.exists(ruta_tramites):
            os.makedirs(ruta_tramites)
            
        loader = DirectoryLoader(ruta_tramites, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        documentos = loader.load()
        
        print(f"📂 Buscando en: {ruta_tramites}")
        print(f"📄 Archivos encontrados: {len(documentos)}")
        
        if len(documentos) == 0:
            print("❌ ALERTA: No se leyó ningún documento.")
            return None
        else:
            print(f"✅ Primer archivo leído: {documentos[0].metadata['source']}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        textos_divididos = text_splitter.split_documents(documentos)
        
        mi_llave = os.getenv("GOOGLE_API_KEY")

        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=mi_llave)
        vectorstore = Chroma.from_documents(documents=textos_divididos, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=mi_llave)
        
        template = """Eres un Consultor Experto del gobierno municipal. Tu objetivo es guiar al ciudadano paso a paso.
        
        REGLAS DE ORO (¡ESTRICTAS!):
        1. LÍMITE DE TEMA: Eres un asistente exclusivo del municipio. SI el ciudadano te pregunta sobre temas fuera de tu alcance, DEBES negarte a responder diciendo: "Lo siento, soy un asistente enfocado exclusivamente en trámites y servicios municipales. ¿En qué proyecto o trámite local te puedo ayudar hoy?".
        2. Usa ÚNICAMENTE la 'Información del municipio' proporcionada abajo. No inventes trámites ni requisitos que no estén ahí. Si la información no menciona el trámite, indícalo claramente.
        3. DIAGNÓSTICO: Haz máximo UNA O DOS preguntas (ej. ¿Vas a construir?, ¿Qué giro es?) para saber qué trámite recomendarle primero, no arrojes toda la información de golpe.
        4. Lee el 'Historial de la conversación' para recordar qué te acaba de responder.

        Historial de la conversación:
        {chat_history}

        Información del municipio (Contexto):
        {context}

        Pregunta actual del ciudadano: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        def formatear_documentos(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        agente_rag = (
            {"context": itemgetter("question") | retriever | formatear_documentos, 
             "question": itemgetter("question"), 
             "chat_history": itemgetter("chat_history")}
            | prompt
            | llm
            | StrOutputParser()
        )
        return agente_rag
    except Exception as e:
        print(f"Error al iniciar el agente: {e}")
        return None
