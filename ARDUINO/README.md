# Firmware de Control - Arduino Nano

Esta carpeta contiene el firmware (`.ino`) para el microcontrolador **Arduino Nano**, encargado de la cinemática y control de actuadores físicos del sistema de clasificación:

1. **Motor Paso a Paso 1:** Accionamiento de tracción para la banda transportadora.
2. **Servomotor:** Deflector/empujador mecánico de componentes hacia el tobogán clasificador.
3. **Motor Paso a Paso 2:** Posicionamiento angular del estuche giratorio de 5 compartimientos según el componente detectado.

### Protocolo de Comunicación Serial
* **Baudrate:** `115200 bps`
* **Entradas (desde Jetson Nano):**
  * `'C'` -> Capacitor electrolítico
  * `'D'` -> DIP switch
  * `'R'` -> Resistencia cerámica
  * `'S'` -> Switch de dos patas
  * `'T'` -> Trimmer
* **Salida (hacia Jetson Nano):**
  * `'k'` -> Carácter de acuse de recibo (ACK) enviado al finalizar la maniobra mecánica de clasificación (fin de carrera/retorno a reposo).
