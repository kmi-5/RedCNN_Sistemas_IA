# 📊 Recursos de Datos para el Laboratorio CNN
# 
# Este archivo centraliza las imágenes procedurales en baja resolución, el motor de interpolación bilineal
# y las matrices de pesos de los Kernels utilizados en el simulador didáctico.
# Es importado de forma modular por `laboratorio_cnn.py` y puede reutilizarse para otros experimentos.

import numpy as np

# =========================================================================
# 1. PLANTILLAS DE IMÁGENES EN BAJA RESOLUCIÓN (16x16)
# =========================================================================

GATO_LOW = [
    [230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230],
    [230, 140, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 140, 230, 230],
    [230, 150, 140, 230, 230, 230, 230, 230, 230, 230, 230, 230, 140, 150, 230, 230],
    [230, 160, 160, 150, 150, 150, 150, 150, 150, 150, 150, 150, 160, 160, 230, 230],
    [180, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 180, 180],
    [180, 180, 180, 180,  30,  30, 180, 180, 180, 180,  30,  30, 180, 180, 180, 180],
    [180, 180, 180, 180,  30,  30, 190, 190, 190, 190,  30,  30, 180, 180, 180, 180],
    [180, 180, 180, 190, 190, 190, 200, 200, 200, 200, 190, 190, 190, 180, 180, 180],
    [180, 180, 180, 190, 190, 210, 150, 150, 150, 210, 190, 190, 180, 180, 180, 180],
    [180, 180, 180, 190, 210, 220, 220, 180, 220, 220, 210, 190, 180, 180, 180, 180],
    [180, 180, 180, 190, 210, 230, 230, 230, 230, 230, 210, 190, 180, 180, 180, 180],
    [180, 180, 180, 190, 190, 210, 220, 220, 220, 210, 190, 190, 180, 180, 180, 180],
    [180, 180, 180, 180, 180, 180, 190, 190, 190, 180, 180, 180, 180, 180, 180, 180],
    [180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180],
    [180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180],
    [180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180]
]

PERRO_LOW = [
    [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
    [240, 160, 160, 240, 240, 240, 240, 240, 240, 240, 240, 240, 160, 160, 240, 240],
    [240, 150, 150, 170, 170, 170, 170, 170, 170, 170, 170, 170, 150, 150, 240, 240],
    [240, 140, 140, 180, 180, 180, 180, 180, 180, 180, 180, 180, 140, 140, 240, 240],
    [240, 140, 140, 190,  30,  30, 190, 190, 190,  30,  30, 190, 140, 140, 240, 240],
    [240, 140, 140, 190,  30,  30, 190, 190, 190,  30,  30, 190, 140, 140, 240, 240],
    [240, 150, 150, 190, 190, 190, 200, 200, 200, 190, 190, 190, 150, 150, 240, 240],
    [240, 160, 160, 190, 190, 210, 220, 220, 220, 210, 190, 190, 160, 160, 240, 240],
    [240, 170, 170, 190, 190, 210,  20,  20,  20, 210, 190, 190, 170, 170, 240, 240],
    [240, 180, 180, 190, 190, 210,  20,  20,  20, 210, 190, 190, 180, 180, 240, 240],
    [240, 180, 180, 190, 190, 210, 210, 210, 210, 210, 190, 190, 180, 180, 240, 240],
    [240, 180, 180, 190, 190, 190, 140, 140, 140, 190, 190, 190, 180, 180, 240, 240],
    [240, 180, 180, 190, 190, 190, 130, 130, 130, 190, 190, 190, 180, 180, 240, 240],
    [240, 180, 180, 190, 190, 190, 190, 190, 190, 190, 190, 190, 180, 180, 240, 240],
    [240, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 240, 240],
    [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240]
]

GEOMETRIA_LOW = [
  [0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0],
  [0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0],
  [0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0],
  [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
  [0,   0,   0, 255,   0,   0, 100, 150, 150, 100,   0, 255,   0,   0,   0,   0],
  [0,   0,   0, 255,   0, 150, 220, 255, 255, 220, 150, 255,   0,   0,   0,   0],
  [0,   0,   0, 255,   0, 200, 255,   0,   0, 255, 200, 255,   0,   0,   0,   0],
  [0,   0,   0, 255,   0, 200, 255,   0,   0, 255, 200, 255,   0,   0,   0,   0],
  [0,   0,   0, 255,   0, 150, 220, 255, 255, 220, 150, 255,   0,   0,   0,   0],
  [0,   0,   0, 255,   0,   0, 100, 150, 150, 100,   0, 255,   0,   0,   0,   0],
  [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
  [0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0],
  [0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0],
  [0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0],
  [0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0],
  [0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0]
]

IMAGE_TEMPLATES = {
    "Gato Selfie 🐈": {
        "description": "Imagen de un gato para estudiar contornos faciales orgánicos.",
        "low": GATO_LOW,
        "focal": [0.55, 0.5, 0.22]
    },
    "Cachorro Alegre 🐕": {
        "description": "Imagen de un perro para detectar formas contrastantes y detalles asimétricos.",
        "low": PERRO_LOW,
        "focal": [0.55, 0.5, 0.25]
    },
    "Patrón Geométrico 📐": {
        "description": "Una mezcla de líneas rectas verticales, horizontales y una elipse central. Ideal para probar filtros de separación de ejes clásicos.",
        "low": GEOMETRIA_LOW,
        "focal": [0.5, 0.5, 0.22]
    }
}

# =========================================================================
# 2. MOTOR DE INTERPOLACIÓN BILINEAL (16x16 -> 64x64)
# =========================================================================

def interpolate_16_to_64(low_res):
    low_res_arr = np.array(low_res, dtype=np.float32)
    size = 64
    matrix = np.zeros((size, size), dtype=np.float32)
    for r in range(size):
        for c in range(size):
            src_r = (r / size) * 16
            src_c = (c / size) * 16
            
            r0 = int(np.floor(src_r))
            r1 = min(r0 + 1, 15)
            c0 = int(np.floor(src_c))
            c1 = min(c0 + 1, 15)
            
            weight_r = src_r - r0
            weight_c = src_c - c0
            
            val00 = low_res_arr[r0, c0]
            val01 = low_res_arr[r0, c1]
            val10 = low_res_arr[r1, c0]
            val11 = low_res_arr[r1, c1]
            
            top = val00 * (1 - weight_c) + val01 * weight_c
            bot = val10 * (1 - weight_c) + val11 * weight_c
            final_val = top * (1 - weight_r) + bot * weight_r
            
            matrix[r, c] = round(final_val)
    return matrix

# =========================================================================
# 3. DICCIONARIO DE FILTROS DE PESOS (KERNELS 3x3)
# =========================================================================

KERNELS = {
  "Identidad": {
    "matrix": np.array([
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0]
    ], dtype=np.float32),
    "description": "Deja la imagen intacta. Sirve de punto de referencia.",
    "analogy": "Un cristal perfectamente templado y transparente."
  },
  "Bordes Horizontales": {
    "matrix": np.array([
      [-1, -2, -1],
      [ 0,  0,  0],
      [ 1,  2,  1]
    ], dtype=np.float32),
    "description": "Detecta cambios abruptos en dirección vertical (gradiente de Y).",
    "analogy": "Una persiana entreabierta que acentúa las sombras del techo y suelo."
  },
  "Bordes Verticales": {
    "matrix": np.array([
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
    ], dtype=np.float32),
    "description": "Detecta transiciones de contraste de izquierda a derecha (Sobel X).",
    "analogy": "La luz rasante del amanecer que choca contra el costado de los muros."
  },
  "Laplaciano (Bordes general)": {
    "matrix": np.array([
      [ 0,  1,  0],
      [ 1, -4,  1],
      [ 0,  1,  0]
    ], dtype=np.float32),
    "description": "Resalta bordes en cualquier sentido midiendo la tasa de cambio local.",
    "analogy": "Un dibujante a lápiz que recorre los contornos extremos con trazo rápido."
  },
  "Enfoque (Sharpen)": {
    "matrix": np.array([
      [ 0, -1,  0],
      [-1,  5, -1],
      [ 0, -1,  0]
    ], dtype=np.float32),
    "description": "Potencia el centro de los píxeles restando la influencia del entorno directo.",
    "analogy": "Una lupa de joyero que realza las micro-grietas y aristas."
  },
  "Desenfoque (Blur)": {
    "matrix": np.array([
      [1/9, 1/9, 1/9],
      [1/9, 1/9, 1/9],
      [1/9, 1/9, 1/9]
    ], dtype=np.float32),
    "description": "Promedia los píxeles vecinos reduciendo imperfecciones nítidas.",
    "analogy": "Mirar el paisaje a través de un vidrio empañado en un día frío."
  },
  "Relieve (Emboss)": {
    "matrix": np.array([
      [-2, -1,  0],
      [-1,  1,  1],
      [ 0,  1,  2]
    ], dtype=np.float32),
    "description": "Crea un efecto tridimensional proyectando sombras asimétricas en diagonal.",
    "analogy": "Modelar una figura presionando una cartulina húmeda desde atrás."
  }
}
