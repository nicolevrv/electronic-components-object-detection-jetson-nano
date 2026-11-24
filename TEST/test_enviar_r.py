#!/usr/bin/env python3
"""
Script de prueba rápida: Enviar 'R' al Arduino Nano desde la Jetson Nano
"""

import serial
import time
import sys

# Puertos más comunes en la Jetson Nano
PUERTOS = ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1']
BAUD_RATE = 115200

def encontrar_arduino():
    for p in PUERTOS:
        try:
            ser = serial.Serial(p, BAUD_RATE, timeout=1)
            return ser, p
        except Exception:
            continue
    return None, None

def main():
    print("==================================================")
    print("  TEST DE COMUNICACIÓN JETSON -> ARDUINO NANO")
    print("==================================================")
    print("Buscando Arduino Nano...")
    
    arduino, puerto = encontrar_arduino()
    
    if not arduino:
        print("\n❌ ERROR: No se encontró ningún Arduino conectado.")
        print("Revisa:")
        print("1. Que el cable USB esté bien conectado a la Jetson.")
        print("2. Ejecuta 'ls /dev/tty*' en la terminal para ver el puerto.")
        print("3. Si da error de permisos, ejecuta: sudo chmod 666 /dev/ttyUSB0")
        sys.exit(1)
        
    print(f"✅ ¡Arduino conectado en: {puerto}!")
    print("Esperando 2 segundos para que el Arduino inicie...")
    time.sleep(2)
    
    print("\n--------------------------------------------------")
    print("Elige un modo:")
    print("1 -> Enviar 'R' una sola vez al presionar ENTER")
    print("2 -> Enviar 'R' automáticamente cada 2 segundos")
    print("--------------------------------------------------")
    
    opcion = input("Selecciona (1 o 2) [Default 1]: ").strip()
    
    if opcion == "2":
        print("\n[MODO AUTOMÁTICO] Enviando 'R' cada 2 segundos. Presiona Ctrl+C para parar.\n")
        contador = 1
        try:
            while True:
                arduino.write(b"R\n")
                arduino.flush()
                print(f"[{contador}] 🚀 Enviada letra 'R'")
                contador += 1
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nDetenido por el usuario.")
    else:
        print("\n[MODO MANUAL] Presiona ENTER para enviar 'R' (escribe 'salir' para terminar):\n")
        while True:
            cmd = input("👉 Presiona ENTER para enviar 'R': ")
            if cmd.lower() == 'salir':
                break
            arduino.write(b"R\n")
            arduino.flush()
            print("🚀 Enviada letra 'R' al Arduino.")
            
    arduino.close()
    print("Puerto cerrado. Fin de la prueba.")

if __name__ == "__main__":
    main()
