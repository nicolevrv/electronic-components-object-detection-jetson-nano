#!/usr/bin/env python3
"""
Detección de Componentes Electrónicos en Tiempo Real con YOLOv8 y TensorRT
NVIDIA Jetson Nano 4GB (JetPack 4.6.1)
(Versión optimizada SIN dependencias de PyCUDA - Usa CUDA Runtime nativo)
"""

import cv2
import numpy as np
import tensorrt as trt
import ctypes
import time

# ---------------- CONFIGURACIÓN ----------------
ENGINE_PATH = "best.engine"
INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
CAMERA_INDEX = 0

# Las 5 clases exactas del proyecto
CLASSES = [
    'capacitor_electrolitico',
    'dip_switch',
    'resistencia_ceramica',
    'switch_de_dos_patas',
    'trimmer'
]

# Colores para dibujar las cajas (formato BGR de OpenCV)
COLORS = [
    (0, 255, 255),  # Amarillo  -> capacitor_electrolitico
    (255, 0, 0),    # Azul      -> dip_switch
    (0, 255, 0),    # Verde     -> resistencia_ceramica
    (0, 165, 255),  # Naranja   -> switch_de_dos_patas
    (255, 0, 255)   # Magenta   -> trimmer
]

# Cargar la librería nativa de CUDA del sistema
try:
    cudart = ctypes.CDLL("libcudart.so")
except Exception:
    cudart = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")

# Constantes de CUDA Memcpy
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2

def cuda_malloc(size_bytes):
    ptr = ctypes.c_void_p()
    status = cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size_bytes))
    if status != 0:
        raise RuntimeError(f"cudaMalloc fallo con codigo: {status}")
    return ptr

def cuda_memcpy(dst, src, size_bytes, kind):
    status = cudart.cudaMemcpy(dst, src, ctypes.c_size_t(size_bytes), ctypes.c_int(kind))
    if status != 0:
        raise RuntimeError(f"cudaMemcpy fallo con codigo: {status}")

class YOLOv8TRT:
    def __init__(self, engine_path, input_size=(640, 640)):
        self.input_size = input_size
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        print(f"[*] Cargando motor TensorRT: {engine_path}...")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        if self.engine is None:
            raise RuntimeError("Error al deserializar el motor TensorRT.")
            
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
            
            # Asignar memoria GPU nativa
            d_ptr = cuda_malloc(nbytes)
            self.bindings.append(d_ptr.value)
            
            # Buffer de memoria en CPU
            host_mem = np.empty(shape, dtype=dtype)
            
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append((d_ptr, nbytes))
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append((d_ptr, nbytes))
                
        print("[+] Motor cargado en memoria GPU de forma nativa.")

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
        
        # 1. Copiar imagen de CPU a GPU
        np.copyto(self.host_inputs[0], blob)
        d_in, in_bytes = self.cuda_inputs[0]
        cuda_memcpy(d_in, self.host_inputs[0].ctypes.data_as(ctypes.c_void_p), in_bytes, CUDA_MEMCPY_HOST_TO_DEVICE)
        
        # 2. Ejecutar inferencia en la GPU
        bindings_arr = (ctypes.c_void_p * len(self.bindings))(*self.bindings)
        self.context.execute_async_v2(bindings=bindings_arr, stream_handle=0)
        cudart.cudaDeviceSynchronize()
        
        # 3. Copiar resultado de GPU a CPU
        d_out, out_bytes = self.cuda_outputs[0]
        cuda_memcpy(self.host_outputs[0].ctypes.data_as(ctypes.c_void_p), d_out, out_bytes, CUDA_MEMCPY_DEVICE_TO_HOST)
        
        # 4. Procesar salida YOLOv8: shape (1, 9, 8400)
        output = self.host_outputs[0][0]
        
        cx = output[0, :]
        cy = output[1, :]
        w = output[2, :]
        h = output[3, :]
        scores = output[4:, :]
        
        max_scores = np.max(scores, axis=0)
        max_classes = np.argmax(scores, axis=0)
        
        mask = max_scores >= CONF_THRESHOLD
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

def main():
    detector = YOLOv8TRT(ENGINE_PATH, input_size=INPUT_SIZE)
    
    print(f"[*] Conectando a la camara USB en indice {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print(f"[!] Error: No se pudo abrir la camara en /dev/video{CAMERA_INDEX}")
        return
        
    print("[+] Camara lista. Presiona 'q' en la ventana para salir.")
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Error al leer fotograma de la camara.")
            break
            
        detections = detector.infer(frame)
        
        for det in detections:
            x, y, w, h = det['box']
            cls_name = det['class_name']
            conf = det['conf']
            color = COLORS[det['class_id']]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
                    
        cv2.imshow("Detector de Componentes - Jetson Nano", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
