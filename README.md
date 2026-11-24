# Proyecto RAN: Sistema Embebido de Detección y Clasificación Automatizada de Componentes Electrónicos

[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Nano%20(4GB)-76B900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-nano)
[![Model](https://img.shields.io/badge/Model-YOLOv8n%20%7C%20TensorRT%20FP16-00FFFF)](https://github.com/ultralytics/ultralytics)
[![Microcontroller](https://img.shields.io/badge/Microcontroller-Arduino%20Nano-00979D?logo=arduino)](https://www.arduino.cc/)
[![Course](https://img.shields.io/badge/UAO-Inteligencia%20Artificial%20Embebida-red)](https://www.uao.edu.co/)

Sistema de visión artificial y control mecatrónico en el borde (*Edge AI*) diseñado para la identificación y clasificación automática de componentes electrónicos en una banda transportadora continua. El despliegue de inferencia se ejecuta enteramente en una **NVIDIA Jetson Nano (4 GB)** optimizado mediante **TensorRT**, comunicándose vía serial con un **Arduino Nano** que comanda tres actuadores mecánicos hacia un estuche giratorio de 5 compartimientos.

---

## 👥 Equipo N.° 6
* **Alessandro Yusty Ceballos** - 2240248
* **Robinson Bastidas Buitrago** - 22235994
* **Nicole Valeria Ruiz Valencia** - 2240429

**Universidad Autónoma de Occidente (UAO)**  
Facultad de Ingeniería – Programas de Ingeniería Mecatrónica e Ingeniería de Datos  
Cali, Colombia (2026)

---

## 🎯 Alineación con los Objetivos de Desarrollo Sostenible (ODS)
* **ODS 12: Producción y Consumo Responsables (Meta 12.5):** Automatización del desensamble, clasificación y valorización de componentes electrónicos para mitigar la generación de chatarra electrónica (*e-waste*).
* **ODS 9: Industria, Innovación e Infraestructura:** Implementación de soluciones de manufactura flexible e inspección de calidad industrial mediante inteligencia artificial embebida de bajo consumo.

---

## 🧩 Clases Detectadas y Mapeo de Actuación
El modelo reconoce 5 clases de componentes electrónicos mediante cajas delimitadoras (*Bounding Boxes*):

| ID | Clase | Carácter Serial | Acción Mecánica (Estuche Giratorio) |
|:--:|:---|:---:|:---|
| `0` | `capacitor_electrolitico` | `'C'` | Posiciona compartimiento 1 + empuje de servo |
| `1` | `dip_switch` | `'D'` | Posiciona compartimiento 2 + empuje de servo |
| `2` | `resistencia_ceramica` | `'R'` | Posiciona compartimiento 3 + empuje de servo |
| `3` | `switch_de_dos_patas` | `'S'` | Posiciona compartimiento 4 + empuje de servo |
| `4` | `trimmer` | `'T'` | Posiciona compartimiento 5 + empuje de servo |

---

## ⚙️ Arquitectura del Sistema y Flujo Lógico

```
[ Cámara Web USB ] 
       │ (Frames 640x480 @ 30 FPS)
       ▼
[ NVIDIA Jetson Nano (4GB) ]
       │ • Pipeline CUDA Nativo (ctypes / libcudart.so)
       │ • Inferencia YOLOv8n optimizado en TensorRT (FP16)
       │ • Control de flujo: Cooldown 8s + Espera de ACK ('k')
       ▼ (UART Serial / USB a 115200 bps)
[ Arduino Nano ]
       │
       ├─► [Motor Paso a Paso 1] -> Tracción continua de la cinta transportadora
       ├─► [Motor Paso a Paso 2] -> Giro angular del estuche (5 compartimientos)
       └─► [Servomotor]          -> Deflexión y empuje de pieza por tobogán
```

### Protocolo de Comunicación y Seguridad (Handshake)
1. La Jetson identifica un componente en la zona de clasificación y transmite el carácter representativo (`'C'`, `'D'`, `'R'`, `'S'`, `'T'`).
2. Se activa una ventana de bloqueo de **8.0 segundos** para permitir el tiempo de traslado cinemático.
3. El Arduino comanda el giro del estuche, empuja el componente mediante el servomotor y retorna a posición de reposo.
4. El Arduino transmite de vuelta el carácter de acuse de recibo **`'k'`**.
5. La Jetson solo autoriza el procesamiento de una nueva pieza cuando se cumplen ambas condiciones: tiempo transcurrido $\ge$ 8 s y recepción confirmada de `'k'`.

---

## 📁 Estructura del Repositorio

```text
PROYECTO EMBEBIDOS/
├── ARDUINO/
│   └── README.md                      # Descripción del firmware y actuadores mecánicos
├── DOCS/
│   ├── Miniproyecto 3.docx             # Rúbrica y enunciado académico
│   └── Paper_IEEE_Proyecto_RAN.pdf    # Paper final documentado en formato IEEE
├── IMÁGENES/
│   └── diagrama_logico.png            # Diagrama de flujo del sistema embebido
├── JETSON/
│   └── clasificador_jetson_arduino.py # Script principal de inferencia TensorRT + Control Serial
├── MODELO/
│   ├── Electronic Component Sorting...zip # Dataset exportado desde Roboflow (YOLOv8 format)
│   └── entrenamiento_yolov8n.ipynb    # Cuaderno de Google Colab para entrenamiento y exportación ONNX
├── TEST/
│   ├── detectar_en_vivo.py            # Prueba de inferencia visual en tiempo real
│   ├── test_teclas_t1_t2_t3.py        # Prueba manual de actuadores por teclado
│   ├── test_enviar_r.py               # Test unitario de transmisión UART
│   ├── TEST_FINAL_R.py                # Test de eco y handshake con Arduino
│   └── clasificador_dipswitch_trimmer.py # Prueba selectiva con dos clases
├── .gitignore                         # Exclusión de archivos binarios y temporales
└── README.md                          # Documentación principal del proyecto
```

---

## 🚀 Despliegue y Ejecución en Jetson Nano

### 1. Requisitos de Software (JetPack 4.6.1)
* Ubuntu 18.04 LTS (aarch64)
* Python 3.6.9
* CUDA 10.2 / cuDNN 8.2 / TensorRT 8.2
* OpenCV 4.1.1 compilado con soporte CUDA

### 2. Conversión a Motor TensorRT
En la Jetson Nano, compilar el archivo `.onnx` (exportado con opset 12) a motor binario optimizado en precisión de media coma flotante (FP16):
```bash
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
```

### 3. Variable de entorno para NumPy (Fix ARMv8)
Si se presenta el error `Illegal instruction (core dumped)` en la Jetson, exportar:
```bash
export OPENBLAS_CORETYPE=ARMV8
```

### 4. Permisos de Puerto Serial y Ejecución
Otorgar permisos de lectura/escritura al puerto del Arduino y ejecutar el script principal:
```bash
sudo chmod 666 /dev/ttyUSB0   # o /dev/ttyACM0 según corresponda
cd JETSON
python3 clasificador_jetson_arduino.py
```

---

## 📊 Métricas de Desempeño del Modelo
* **Parámetros:** ~3.0 Millones (YOLOv8 Nano)
* **Dataset:** 139 imágenes (97 entrenamiento, 28 validación, 14 prueba)
* **mAP@50 (Test):** 0.995 (99.5%)
* **mAP@50-95 (Test):** 0.909 (90.9%)
* **Velocidad de Inferencia:** ~13 FPS en GPU NVIDIA Maxwell (128 CUDA cores)
