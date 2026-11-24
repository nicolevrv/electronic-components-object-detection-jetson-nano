#!/usr/bin/env python3
import glob
import sys
import time
import serial

ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
if not ports:
    print("No se encontró Arduino en /dev/ttyUSB* o /dev/ttyACM*")
    sys.exit(1)

port = ports[0]
print(f"Abriendo {port} a 115200 baudios...")

with serial.Serial(port, 115200, timeout=0.5) as ser:
    time.sleep(2.0)  # Esperar reinicio del bootloader del Arduino al abrir puerto
    ser.reset_input_buffer()
    print("Enviando 'R' cada 2 segundos (Ctrl+C para salir)...\n")

    try:
        while True:
            ser.write(b"R")
            print("JETSON TX -> Enviado 'R'")

            ack = ser.read(1)
            if ack == b"R":
                print("ARDUINO ECO -> Confirmado 'R' (LED D13 parpadeó)\n")
            else:
                print(f"SIN ECO / INESPERADO -> {ack!r}\n")

            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nPuerto cerrado.")