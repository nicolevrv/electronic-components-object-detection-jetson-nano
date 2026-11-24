#!/usr/bin/env python3
"""
=============================================================================
SISTEMA DE CLASIFICACIÓN DE COMPONENTES ELECTRÓNICOS (5 CLASES)
NVIDIA Jetson Nano 4GB (JetPack 4.6.1) + Arduino Nano
=============================================================================
Mapeo de Clases:
  'T' -> Trimmer
  'S' -> Switch de dos patas
  'D' -> DipSwitch
  'C' -> Capacitor electrolítico
  'R' -> Resistencia cerámica

Control de Flujo Estricto:
  1. Envía la señal al Arduino.
  2. BLOQUEO OBLIGATORIO DE MÍNIMO 8.0 SEGUNDOS entre una señal y la siguiente.
  3. Espera la confirmación 'k' de vuelta del Arduino (fin de carrera).
  4. Ambas condiciones (8s transcurridos Y 'k' recibido) son indispensables
     antes de autorizar la siguiente pieza.
"""

import cv2
import numpy as np
import tensorrt as trt
import ctypes
import serial
import time
import os

# ==================== CONFIGURACIÓN ====================
ENGINE_PATH = "best.engine"
INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45
CAMERA_INDEX = 0

POSIBLES_PUERTOS_ARDUINO = ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1']
BAUD_RATE = 115200

# ⏱️ TIEMPO MÍNIMO ESTRICTO ENTRE SEÑALES (8 SEGUNDOS)
DELAY_ESTRICTO_SEGUNDOS = 8.0

# ----------------- LAS 5 CLASES Y SU MAPEO -----------------
CLASSES = [
    'capacitor_electrolitico', # ID 0 -> 'C'
    'dip_switch',              # ID 1 -> 'D'
    'resistencia_ceramica',    # ID 2 -> 'R'
    'switch_de_dos_patas',     # ID 3 -> 'S'
    'trimmer'                  # ID 4 -> 'T'
]

MAPEO_SERIAL = {
    'trimmer': 'T',
    'switch_de_dos_patas': 'S',
    'dip_switch': 'D',
    'capacitor_electrolitico': 'C',
    'resistencia_ceramica': 'R'
}

COLORS = [
    (0, 255, 255),  # Amarillo -> Capacitor ('C')
    (255, 0, 0),    # Azul     -> DipSwitch ('D')
    (0, 255, 0),    # Verde    -> Resistencia ('R')
    (0, 165, 255),  # Naranja  -> Switch ('S')
    (255, 0, 255)   # Magenta  -> Trimmer ('T')
]

# ==================== DRIVER CUDA 64-BIT (CTYPES) ====================
try:
    cudart = ctypes.CDLL("libcudart.so")
except Exception:
    cudart = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2

cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaMalloc.restype = ctypes.c_int
cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
cudart.cudaMemcpy.restype = ctypes.c_int
cudart.cudaDeviceSynchronize.restype = ctypes.c_int

def cuda_malloc(size_bytes):
    ptr = ctypes.c_void_p()
    status = cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size_bytes))
    if status != 0:
        raise RuntimeError(f"cudaMalloc fallo: {status}")
    return ptr

def cuda_memcpy(dst, src, size_bytes, kind):
    status = cudart.cudaMemcpy(dst, src, ctypes.c_size_t(size_bytes), ctypes.c_int(kind))
    if status != 0:
        raise RuntimeError(f"cudaMemcpy fallo: {status}")

# ==================== DETECTOR TENSORRT ====================
class YOLOv8TRT:
    def __init__(self, engine_path, input_size=(640, 640)):
        self.input_size = input_size
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"No se encontro el archivo de motor: {engine_path}")
            
        print(f"[*] Cargando motor TensorRT: {engine_path}...")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.bindings = []
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = trt.volume(shape)
            nbytes = size * np.dtype(dtype).itemsize
            
            d_ptr = cuda_malloc(nbytes)
            self.bindings.append(d_ptr.value)
            host_mem = np.empty(shape, dtype=dtype)
            
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append((d_ptr, nbytes))
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append((d_ptr, nbytes))
                
        print(f"[+] Motor TensorRT listo en la GPU.")

    def preprocess(self, img):
        h, w = img.shape[:2]
        resized = cv2.resize(img, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = np.ascontiguousarray(rgb.astype(np.float32) / 255.0)
        chw = np.transpose(normalized, (2, 0, 1))
        batch = np.ascontiguousarray(np.expand_dims(chw, axis=0))
        return batch, w, h

    def infer(self, frame):
        blob, orig_w, orig_h = self.preprocess(frame)
        
        # CPU -> GPU
        np.copyto(self.host_inputs[0], blob)
        d_in, in_bytes = self.cuda_inputs[0]
        cuda_memcpy(d_in, self.host_inputs[0].ctypes.data_as(ctypes.c_void_p), in_bytes, CUDA_MEMCPY_HOST_TO_DEVICE)
        
        # Inferencia
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=0)
        cudart.cudaDeviceSynchronize()
        
        # GPU -> CPU
        d_out, out_bytes = self.cuda_outputs[0]
        cuda_memcpy(self.host_outputs[0].ctypes.data_as(ctypes.c_void_p), d_out, out_bytes, CUDA_MEMCPY_DEVICE_TO_HOST)
        
        # Parsear salida
        output = self.host_outputs[0][0]
        cx, cy, w, h = output[0, :], output[1, :], output[2, :], output[3, :]
        scores = output[4:, :]
        
        max_scores = np.max(scores, axis=0)
        max_classes = np.argmax(scores, axis=0)
        
        mask = max_scores >= CONF_THRESHOLD
        if not np.any(mask):
            return []
            
        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        max_scores = max_scores[mask]
        max_classes = max_classes[mask]
        
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        
        boxes = []
        confidences = []
        class_ids = []
        
        for i in range(len(max_scores)):
            x1 = int((cx[i] - w[i] / 2) * scale_x)
            y1 = int((cy[i] - h[i] / 2) * scale_y)
            box_w = int(w[i] * scale_x)
            box_h = int(h[i] * scale_y)
            boxes.append([x1, y1, box_w, box_h])
            confidences.append(float(max_scores[i]))
            class_ids.append(int(max_classes[i]))
            
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)
        
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                results.append({
                    'class_id': class_ids[idx],
                    'class_name': CLASSES[class_ids[idx]],
                    'conf': confidences[idx],
                    'box': boxes[idx]
                })
        return results

# ==================== CONEXIÓN SERIAL ====================
def conectar_arduino():
    for puerto in POSIBLES_PUERTOS_ARDUINO:
        try:
            ser = serial.Serial(puerto, BAUD_RATE, timeout=0.01, write_timeout=0.05)
            time.sleep(1.5)
            ser.reset_input_buffer()
            print(f"[+] Conectado al Arduino en: {puerto}")
            return ser
        except Exception:
            continue
    print("[!] Arduino no detectado. Corriendo en modo visual.")
    return None

# ==================== BUCLE PRINCIPAL ====================
def main():
    arduino = conectar_arduino()
    detector = YOLOv8TRT(ENGINE_PATH, input_size=INPUT_SIZE)
    
    print(f"[*] Abriendo camara USB en /dev/video{CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"[!] Error: No se pudo abrir la camara /dev/video{CAMERA_INDEX}.")
        return

    print("==============================================================")
    print("  SISTEMA CLASIFICADOR (DELAY ESTRICTO: 8s + ACK 'k')         ")
    print("  Presiona 'q' en la ventana para salir.                      ")
    print("==============================================================")

    # Variables de control de tiempo y estado
    tiempo_desbloqueo = 0.0          # Timestamp hasta cuando está prohibido enviar otra señal
    esperando_confirmacion_k = False # True mientras el servo se mueve y no ha llegado 'k'
    ultimo_componente_enviado = ""
    ultima_letra_enviada = ""
    fps_smooth = 0.0

    while True:
        t_inicio = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("[!] Error leyendo fotograma.")
            break

        tiempo_actual = time.time()
        
        # 1. LEER RESPUESTAS DEL ARDUINO ('k')
        if arduino and arduino.is_open and arduino.in_waiting > 0:
            try:
                datos = arduino.read_all().decode('utf-8', errors='ignore')
                if 'k' in datos:
                    esperando_confirmacion_k = False
                    print("✅ [ACK RECIBIDO] El servomotor volvio a reposo ('k').")
            except Exception as e:
                print(f"[!] Error leyendo serial: {e}")

        # 2. EVALUAR SI EL SISTEMA ESTÁ 100% LIBRE PARA ENVIAR
        # Condición 1: Ya transcurrieron los 8 segundos estrictos desde el último envío
        tiempo_restante = max(0.0, tiempo_desbloqueo - tiempo_actual)
        delay_cumplido = (tiempo_restante <= 0.0)
        
        # Condición 2: El Arduino ya envió 'k' confirmando fin de carrera
        puede_enviar_nueva_pieza = delay_cumplido and (not esperando_confirmacion_k)

        # 3. INFERENCIA
        detections = detector.infer(frame)
        componente_para_enviar = None

        for det in detections:
            x, y, w, h = det['box']
            cls_name = det['class_name']
            conf = det['conf']
            color = COLORS[det['class_id']]
            letra = MAPEO_SERIAL[cls_name]

            etiqueta = f"[{letra}] {cls_name} ({conf:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, etiqueta, (x, max(25, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if puede_enviar_nueva_pieza and componente_para_enviar is None:
                componente_para_enviar = cls_name

        # 4. DISPARAR ORDEN SERIAL SI EL SISTEMA ESTÁ LIBRE
        if componente_para_enviar and puede_enviar_nueva_pieza:
            caracter = MAPEO_SERIAL[componente_para_enviar]
            if arduino and arduino.is_open:
                try:
                    arduino.reset_input_buffer()
                    arduino.write(f"{caracter}\n".encode('utf-8'))
                    arduino.flush()
                except Exception as e:
                    print(f"[!] Error enviando serial: {e}")

            # Activar bloqueo estricto de 8 segundos A PARTIR DE AHORA
            tiempo_desbloqueo = tiempo_actual + DELAY_ESTRICTO_SEGUNDOS
            esperando_confirmacion_k = True
            ultimo_componente_enviado = componente_para_enviar
            ultima_letra_enviada = caracter
            
            hora_actual = time.strftime("%H:%M:%S")
            print(f"\n🚀 [{hora_actual}] SEÑAL ENVIADA: '{caracter}' ({componente_para_enviar})")
            print(f"    🔒 SISTEMA BLOQUEADO durante 8.0 segundos exactos...")

        # 5. CÁLCULO DE FPS REAL
        t_delta = time.time() - t_inicio
        fps_instantaneo = 1.0 / t_delta if t_delta > 0 else 0
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps_instantaneo if fps_smooth > 0 else fps_instantaneo

        cv2.putText(frame, f"FPS: {int(fps_smooth)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

        # 6. BANNER VISUAL SUPERIOR
        if not puede_enviar_nueva_pieza:
            # Mostrar la cuenta regresiva exacta de los 8 segundos
            if tiempo_restante > 0.0:
                texto_banner = f"DELAY BLOQUEADO [{ultima_letra_enviada}] - Espera: {int(tiempo_restante) + 1}s"
                color_banner = (0, 100, 220) # Naranja oscuro
            else:
                texto_banner = f"ESPERANDO CONFIRMACION 'k' DEL SERVO..."
                color_banner = (0, 70, 180)  # Naranja-rojo

            cv2.rectangle(frame, (10, 50), (630, 95), color_banner, -1)
            cv2.putText(frame, texto_banner, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        else:
            # Listo para clasificar
            cv2.rectangle(frame, (10, 50), (520, 95), (0, 150, 0), -1)
            cv2.putText(frame, "ESTADO: LISTO PARA CLASIFICAR", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        cv2.imshow("Clasificador Jetson Nano - 5 Clases", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if arduino and arduino.is_open:
        arduino.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
