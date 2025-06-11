import os                      # Importamos para manipulación de rutas.
import random                  # Importamos para generar colores aleatorios 

import cv2                     # Importamos OpenCV para manejar video e imágenes.
from ultralytics import YOLO  # Importamos el modelo YOLOv8 desde la librería Ultralytics.
from tracker import Tracker    # Importamos nuestra clase de seguimiento personalizada.

# Definimos el nombre del video base 
name_video = 'sprint'

# Construimos la ruta al archivo de video de entrada
video_path = os.path.join('.', 'data', name_video + '.mp4')

# Construimos la ruta donde guardaremos el video de salida procesado
video_out_path = os.path.join('.', 'data_out', name_video + '_out.mp4')

# Abrimos el video de entrada
cap = cv2.VideoCapture(video_path)

# Leemos el primer frame del video
ret, frame = cap.read()

# Creamos el objeto para escribir el video de salida con las mismas dimensiones y FPS que el original
cap_out = cv2.VideoWriter(
    video_out_path,                             # Ruta de salida
    cv2.VideoWriter_fourcc(*'MP4V'),            # Codec MP4
    cap.get(cv2.CAP_PROP_FPS),                  # Leemos la tasa de cuadros del video original
    (frame.shape[1], frame.shape[0])            # Dimensiones del frame: ancho x alto
)

# Cargamos el modelo YOLOv8 versión nano (mas ligero y rápido)
model = YOLO("yolov8n.pt")

# Inicializamos el objeto tracker, que nos permitirá dar seguimiento a los objetos detectados
tracker = Tracker()

# Generamos una lista de 10 colores aleatorios (RGB) para identificar visualmente distintos objetos
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# Definimos un umbral de confianza: solo consideraremos detecciones con probabilidad mayor a este valor
detection_threshold = 0.5

# Iniciamos el ciclo principal: seguimos mientras podamos leer frames del video
while ret:

    # Aplicamos el modelo YOLO al frame actual para detectar objetos
    results = model(frame)

    # Iteramos sobre los resultados obtenidos del modelo
    for result in results:
        detections = []  # Inicializamos la lista de detecciones válidas para este frame

        # Iteramos sobre todas las cajas detectadas
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r  # Extraemos las coordenadas y el puntaje de cada caja

            # Convertimos las coordenadas a enteros
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)

            # Filtramos las detecciones según el umbral de confianza
            # and class_id == 0 para que solo detecte personas
            if score > detection_threshold and class_id == 0:
                detections.append([x1, y1, x2, y2, score])  # Guardamos las detecciones válidas

        # Actualizamos el tracker con el frame actual y las detecciones válidas
        tracker.update(frame, detections)

        # Recorremos las trayectorias activas para dibujar las cajas en el frame
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (colors[track_id % len(colors)]),
                3
            )

            cv2.putText(
                frame,
                f'ID {track_id}',
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                colors[track_id % len(colors)],
                2,
                cv2.LINE_AA
            )


    # Escribimos el frame procesado (con cajas y seguimiento) en el video de salida
    cap_out.write(frame)

    # Si quisiéramos visualizar en tiempo real, descomentamos estas líneas
    # Solo es recomendado si la computadora tiene gran capacidad
    # cv2.imshow('frame', frame)
    # cv2.waitKey(25)

    # Leemos el siguiente frame del video
    ret, frame = cap.read()

# Liberamos los recursos: cerramos el video de entrada y el de salida
cap.release()
cap_out.release()

# Si activamos la visualización, también cerramos las ventanas de OpenCV
# cv2.destroyAllWindows()