import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Cargar llave secreta
load_dotenv()

print("1. Despertando al agente y leyendo el trámite...")
loader = TextLoader('tramite_licencia.txt', encoding='utf-8')
documentos = loader.load()

print("2. Creando la memoria vectorial...")
# ¡LA SOLUCIÓN! Usamos el modelo global más reciente de Google que reemplazó a todos los anteriores
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma.from_documents(documents=documentos, embedding=embeddings)
retriever = vectorstore.as_retriever()

print("3. Conectando con el cerebro de Gemini...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 4. Plantilla de Personalidad (System Prompt)
template = """Eres un asistente amable del gobierno municipal.
Usa ÚNICAMENTE la siguiente información para responder a la pregunta del ciudadano.
Si la respuesta no está en el texto de abajo, di amablemente que no tienes esa información. No inventes nada.

Información del municipio:
{context}

Pregunta del ciudadano: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Función para extraer el texto de los documentos
def formatear_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("4. Uniendo las piezas (Memoria + Reglas + Cerebro)...")
agente_rag = (
    {"context": retriever | formatear_documentos, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("-" * 50)
pregunta_ciudadano = "Hola, quiero abrir un local, ¿cuánto cuesta la licencia y en cuánto tiempo me responden?"
print(f"Ciudadano: {pregunta_ciudadano}\n")

print("Agente pensando y buscando en los documentos...")
# 5. Ejecutamos la cadena
respuesta = agente_rag.invoke(pregunta_ciudadano)

print(f"\nAgente: {respuesta}")
print("-" * 50)