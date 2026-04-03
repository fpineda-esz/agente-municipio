import streamlit as st
import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from operator import itemgetter
# --- NUEVAS IMPORTACIONES ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def guardar_log(pregunta, respuesta):
    # Nombre del archivo donde se guardará todo
    archivo_csv = "registro_consultas.csv"
    
    # Verificamos si el archivo ya existe para saber si ponemos los encabezados
    existe = os.path.isfile(archivo_csv)
    
    # Abrimos el archivo en modo "a" (append/agregar) para no borrar lo anterior
    with open(archivo_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Si es la primera vez que se crea, le ponemos los títulos a las columnas
        if not existe:
            writer.writerow(["Fecha_y_Hora", "Pregunta_del_Ciudadano", "Respuesta_del_Bot"])
        
        # Capturamos la hora exacta y guardamos la fila
        hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([hora_actual, pregunta, respuesta])

# 1. Configuración de la página
st.set_page_config(page_title="Asistente Municipal", page_icon="🏛️", layout="centered")

# 2. Cargar llave secreta
load_dotenv()

# --- INICIO DEL BACKEND ---
@st.cache_resource
def iniciar_agente():
    try:
        # ACTUALIZACIÓN 1: Leer carpeta completa. 
        # Asegúrate de crear una carpeta llamada 'tramites' junto a este archivo.
        loader = DirectoryLoader('./tramites', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        documentos = loader.load()
        
        # ACTUALIZACIÓN 2: Partir los documentos en pedazos para no saturar a la IA
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        textos_divididos = text_splitter.split_documents(documentos)
        
        # Crear la memoria vectorial con los pedacitos
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        vectorstore = Chroma.from_documents(documents=textos_divididos, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # Trae los 4 fragmentos más relevantes
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        # ACTUALIZACIÓN 3: El Nuevo "Súper Prompt" Consultor con Guardrails
        template = """Eres un Consultor Experto del gobierno municipal. Tu objetivo es guiar al ciudadano paso a paso.
        
        REGLAS DE ORO (¡ESTRICTAS!):
        1. LÍMITE DE TEMA: Eres un asistente exclusivo del municipio. SI el ciudadano te pregunta sobre temas fuera de tu alcance (ej. recetas de cocina, clima, política nacional, tareas escolares, chistes, etc.), DEBES negarte a responder de forma amable y estandarizada diciendo: "Lo siento, soy un asistente enfocado exclusivamente en trámites y servicios municipales. ¿En qué proyecto o trámite local te puedo ayudar hoy?".
        2. Usa ÚNICAMENTE la 'Información del municipio' proporcionada abajo. No inventes trámites ni requisitos que no estén ahí. Si la información no menciona el trámite, indícalo claramente.
        3. DIAGNÓSTICO: Si el ciudadano te dice un objetivo general (ej. "quiero abrir un negocio"), no le arrojes toda la información de golpe. Hazle máximo UNA O DOS preguntas (ej. ¿Vas a construir?, ¿Qué giro es?) para saber qué trámite recomendarle primero.
        4. Lee el 'Historial de la conversación' para recordar qué te acaba de responder el ciudadano y no volver a preguntarle lo mismo.

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
             "chat_history": itemgetter("chat_history")
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return agente_rag
    except Exception as e:
        st.error(f"Error al iniciar el agente: {e}")
        return None

agente = iniciar_agente()
# --- FIN DEL BACKEND ---

# --- INICIO DEL FRONTEND ---
st.title("🏛️ EasyBot.GOV - Tu Agente de Trámites y Servicios")
st.markdown("Hola, soy tu agente virtual que te acompañará para encontrar el trámite o servicio que buscas. Pregúntame o cuéntame qué es lo que quieres hacer y te daré una orientación clara y concisa.")

# MEJORA 1: Menú lateral (Sidebar) con botón para limpiar el chat
with st.sidebar:
    st.title("⚙️ Opciones")
    st.markdown("Usa este botón si quieres borrar el historial y empezar un tema nuevo.")
    if st.button("🗑️ Nueva Consulta"):
        st.session_state.mensajes = []
        st.rerun() # Reinicia la interfaz para limpiar la pantalla

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# MEJORA 2: Botones de opciones rápidas (solo se muestran si el chat está vacío)
pregunta_rapida = None
if len(st.session_state.mensajes) == 0:
    st.markdown("💡 **Puedes escribir tu duda abajo o elegir una opción rápida:**")
    col1, col2 = st.columns(2)
    # Si el usuario hace clic, guardamos la pregunta en la variable
    if col1.button("🏢 ¿Cómo tramito un Uso de Suelo?"):
        pregunta_rapida = "¿Cómo tramito un Uso de Suelo?"
    if col2.button("🏪 Requisitos para abrir un negocio"):
        pregunta_rapida = "Quiero abrir un negocio, ¿qué requisitos necesito?"
    st.markdown("---") # Una línea divisoria visual

def formatear_historial(mensajes):
    historial_texto = ""
    for msg in mensajes[-6:]: # Solo recordamos los últimos 6 mensajes
        rol = "Ciudadano" if msg["rol"] == "user" else "Agente"
        historial_texto += f"{rol}: {msg['contenido']}\n"
    return historial_texto

# Renderizamos los mensajes guardados
for i, mensaje in enumerate(st.session_state.mensajes):
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])
        
        # MEJORA 4: Sistema de Feedback (solo para las respuestas del bot)
        if mensaje["rol"] == "assistant":
            # st.feedback crea los íconos interactivos. Necesitamos un 'key' único por cada mensaje.
            valoracion = st.feedback("thumbs", key=f"feedback_{i}")
            
            # Si el usuario ya presionó un botón, Streamlit devuelve 1 (arriba) o 0 (abajo)
            if valoracion is not None:
                if valoracion == 1:
                    st.caption("¡Gracias! 👍 Me alegra haberte ayudado.")
                elif valoracion == 0:
                    st.caption("Gracias por avisar. 👎 Tomaremos nota para mejorar esta ficha.")

# Capturamos lo que el usuario escriba en la barra de texto
pregunta_usuario = st.chat_input("Escribe tu duda o proyecto aquí...")

# Unificamos ambas fuentes de entrada (lo que tecleó o el botón que presionó)
consulta_final = pregunta_usuario or pregunta_rapida

if consulta_final:
    with st.chat_message("user"):
        st.markdown(consulta_final)
    
    # Extraemos el historial antes de agregar la nueva pregunta
    historial_actual = formatear_historial(st.session_state.mensajes)
    st.session_state.mensajes.append({"rol": "user", "contenido": consulta_final})

    with st.chat_message("assistant"):
        with st.spinner("Analizando tu caso en los reglamentos..."):
            if agente:
                respuesta = agente.invoke({
                    "question": consulta_final,
                    "chat_history": historial_actual
                })
            else:
                respuesta = "Aún no tengo documentos en mi base de datos."
            st.markdown(respuesta)
    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
    # --- NUEVA LÍNEA: Guardamos la interacción en el Excel (CSV) ---
    guardar_log(consulta_final, respuesta)
    st.rerun()
# --- FIN DEL FRONTEND ---