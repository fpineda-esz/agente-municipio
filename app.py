import streamlit as st
import os
from dotenv import load_dotenv

# --- IMPORTAMOS NUESTROS MÓDULOS ---
from auditoria import guardar_log
from motor_ia import crear_agente

# 1. Configuración de la página
st.set_page_config(page_title="Asistente Municipal", page_icon="🏛️", layout="centered")

# --- CSS VISUAL ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 0rem; }
    [data-testid="stChatMessage"]:has(.marca-ciudadano) { background-color: #2b303b; border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    [data-testid="stChatMessage"]:has(.marca-agente) { background-color: transparent; border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    div.stButton > button:first-child { border-radius: 8px; border: 1px solid #c8a14d; color: #ffffff; transition: all 0.3s; }
    div.stButton > button:first-child:hover { background-color: #c8a14d; border-color: #c8a14d; color: #000000; }
</style>
""", unsafe_allow_html=True)

# 2. Verificación de Llave
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ ERROR: No se encontró la llave 'GOOGLE_API_KEY' en el archivo .env")

# 3. Iniciar el Agente (Guardado en memoria de sesión)
if "agente" not in st.session_state:
    with st.spinner("Conectando con la base de datos municipal..."):
        st.session_state.agente = crear_agente()

# --- INTERFAZ ---
st.title("🏛️ EasyBot.GOV - Tu Agente de Trámites y Servicios")
st.markdown("Hola, soy tu agente virtual que te acompañará para encontrar el trámite o servicio que buscas. Pregúntame o cuéntame qué es lo que quieres hacer y te daré una orientación clara y concisa.")

with st.sidebar:
    st.title("⚙️ Opciones")
    if st.button("🗑️ Nueva Consulta"):
        st.session_state.mensajes = []
        st.rerun()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

pregunta_rapida = None
if len(st.session_state.mensajes) == 0:
    st.markdown("💡 **Puedes escribir tu duda abajo o elegir una opción rápida:**")
    col1, col2 = st.columns(2)
    if col1.button("🏢 ¿Cómo tramito un Permiso de Uso de Suelo?"):
        pregunta_rapida = "¿Cómo tramito un Permiso de Uso de Suelo?"
    if col2.button("🏪 Requisitos para abrir un negocio"):
        pregunta_rapida = "Quiero abrir un negocio, ¿qué requisitos necesito?"
    st.markdown("---")

def formatear_historial(mensajes):
    historial_texto = ""
    for msg in mensajes[-6:]:
        rol = "Ciudadano" if msg["rol"] == "user" else "Agente"
        historial_texto += f"{rol}: {msg['contenido']}\n"
    return historial_texto

for i, mensaje in enumerate(st.session_state.mensajes):
    with st.chat_message(mensaje["rol"]):
        if mensaje["rol"] == "user":
            st.markdown(f"<span class='marca-ciudadano'></span> {mensaje['contenido']}", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='marca-agente'></span> {mensaje['contenido']}", unsafe_allow_html=True)
            valoracion = st.feedback("thumbs", key=f"feedback_{i}")
            if valoracion is not None:
                st.caption("¡Gracias por tu retroalimentación!" if valoracion == 1 else "Gracias por avisar. Tomaremos nota.")

pregunta_usuario = st.chat_input("Escribe tu duda o proyecto aquí...")
consulta_final = pregunta_usuario or pregunta_rapida

if consulta_final:
    with st.chat_message("user"):
        st.markdown(f"<span class='marca-ciudadano'></span> {consulta_final}", unsafe_allow_html=True)
    
    historial_actual = formatear_historial(st.session_state.mensajes)
    st.session_state.mensajes.append({"rol": "user", "contenido": consulta_final})

    with st.chat_message("assistant"):
        with st.spinner("Analizando tu caso en los reglamentos..."):
            if st.session_state.agente:
                respuesta = st.session_state.agente.invoke({"question": consulta_final, "chat_history": historial_actual})
            else:
                respuesta = "Aún no tengo documentos en mi base de datos."
            st.markdown(f"<span class='marca-agente'></span> {respuesta}", unsafe_allow_html=True)
            
    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
    guardar_log(consulta_final, respuesta) # Mandamos llamar nuestra función de auditoria.py
    st.rerun()
