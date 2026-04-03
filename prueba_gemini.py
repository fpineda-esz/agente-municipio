import os
from dotenv import load_dotenv
from google import genai

# 1. Cargamos tu llave secreta
load_dotenv()

# 2. Inicializamos la conexión con la librería nueva
cliente = genai.Client()

print("--- Conectando con Gemini (Nueva API) ---")
# 3. Le mandamos el mensaje usando el modelo más reciente
respuesta = cliente.models.generate_content(
    model='gemini-2.5-flash',
    contents="Hola Gemini, dime en una sola oración corta que nuestra conexión de prueba desde la Mac de Franco fue un éxito usando la nueva API."
)

print(f"Respuesta de la IA: {respuesta.text}")
print("-----------------------------------------")