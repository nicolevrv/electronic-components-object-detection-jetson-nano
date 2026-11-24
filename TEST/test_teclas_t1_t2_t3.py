#!/usr/bin/env python3
"""
Control Manual por Teclado: Jetson Nano -> Arduino Nano
Envía 't1', 't2' o 't3' al presionar una sola tecla (sin necesidad de dar ENTER).
Compatible con Linux (Jetson Nano) y Windows.
"""

import serial
import time
import sys
import os

# Puertos típicos en la Jetson Nano / Linux
PUERTOS = ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1']
BAUD_RATE = 115200

# ----------------- DETECCIÓN DE PULSACIÓN DE TECLA -----------------
# Permite detectar teclas al instante sin necesidad de presionar ENTER
def get_key():
    if os.name == 'nt':  # Windows
        import msvcrt
        ch = msvcrt.getch()
        try:
            return ch.decode('utf-8')
        except UnicodeDecodeError:
            return ''
    else:  # Linux / Jetson Nano
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

# ----------------- CONEXIÓN CON ARDUINO -----------------
def encontrar_arduino():
    for p in PUERTOS:
        try:
            ser = serial.Serial(p, BAUD_RATE, timeout=0.1)
            return ser, p
        except Exception:
            continue
    return None, None

def main():
    print("==================================================")
    print("   CONTROL MANUAL POR TECLADO: JETSON -> ARDUINO  ")
    print("==================================================")
    print("Buscando Arduino Nano...")
    
    arduino, puerto = encontrar_arduino()
    
    if not arduino:
        print("\n❌ ERROR: No se encontró ningún Arduino conectado.")
        print("Revisa la conexión USB y los permisos (sudo chmod 666 /dev/ttyUSB0).")
        sys.exit(1)
        
    print(f"✅ ¡Arduino conectado en: {puerto}!")
    print("Esperando 2 segundos para estabilizar la conexión...")
    time.sleep(2)

    print("\n--------------------------------------------------")
    print("  PRESIONA UNA TECLA DIRECTAMENTE:")
    print("   [1] -> Enviar 't1'")
    print("   [2] -> Enviar 't2'")
    print("   [3] -> Enviar 't3'")
    print("   [q] o [ESC] -> Salir del programa")
    print("--------------------------------------------------\n")

    # Mapeo de teclas a comandos
    MAPEO = {
        '1': 't1',
        '2': 't2',
        '3': 't3'
    }

    try:
        while True:
            tecla = get_key()

            # Salir con 'q' o tecla ESC (código \x1b)
            if tecla.lower() == 'q' or tecla == '\x1b':
                print("\nSaliendo del programa...")
                break

            if tecla in MAPEO:
                comando = MAPEO[tecla]
                mensaje = f"{comando}\n"
                arduino.write(mensaje.encode('utf-8'))
                arduino.flush()
                print(f"👉 Tecla presionada: [{tecla}]  ===>  🚀 Enviado a Arduino: \"{comando}\"")
            else:
                print(f"Tecla [{tecla}] no asignada. Usa 1, 2 o 3 (o 'q' para salir).")

    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario.")

    finally:
        if arduino and arduino.is_open:
            arduino.close()
        print("Puerto serial cerrado correctamente.")

if __name__ == "__main__":
    main()
