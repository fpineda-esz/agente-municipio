# Abrimos el archivo en modo lectura ('r') y le indicamos que use 'utf-8' para no tener problemas con los acentos.
with open('tramite_licencia.txt', 'r', encoding='utf-8') as archivo:
    contenido = archivo.read()
    print("--- LEYENDO BASE DE DATOS DEL MUNICIPIO ---")
    print(contenido)