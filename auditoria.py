import os
import csv
from datetime import datetime

def guardar_log(pregunta, respuesta):
    # Nombre del archivo donde se guardará todo
    archivo_csv = "registro_consultas.csv"
    
    # Verificamos si el archivo ya existe para saber si ponemos los encabezados
    existe = os.path.isfile(archivo_csv)
    
    # Abrimos el archivo en modo "a" (append/agregar)
    with open(archivo_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Si es la primera vez que se crea, le ponemos los títulos
        if not existe:
            writer.writerow(["Fecha_y_Hora", "Pregunta_del_Ciudadano", "Respuesta_del_Bot"])
        
        # Capturamos la hora exacta y guardamos la fila
        hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([hora_actual, pregunta, respuesta])