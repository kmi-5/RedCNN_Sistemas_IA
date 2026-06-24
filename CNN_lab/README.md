# 🔬 Laboratorio CNN Didáctico & Interactivo

Un simulador interactivo y módulo de *storytelling* portátil desarrollado con **Streamlit**, **Python** y **Plotly** para comprender el funcionamiento interno de las **Redes Neuronales Convolucionales (CNN)**.

Este proyecto permite visualizar "el viaje del píxel": desde una cuadrícula numérica bidimensional cruda hasta la extracción avanzada de características espaciales por medio de convoluciones, funciones de activación y submuestreos.

---

## 🚀 Instrucciones de Instalación y Uso

Sigue estos sencillos pasos para ejecutar el laboratorio de forma portable, offline y local en tu computadora:

### 1. Requisitos Previos
Asegúrate de tener instalado **Python 3.8 o superior**. Puedes descargarlo desde [python.org](https://www.python.org/).

### 2. Instalación de Dependencias
Abre una terminal o consola de comandos en la carpeta raíz de este proyecto y ejecuta:

```bash
pip install -r requirements.txt
```

*Nota: Esto instalará automáticamente `streamlit`, `numpy`, `pandas`, `Pillow`, `matplotlib` y `plotly`.*

### 3. Ejecución del Laboratorio
Inicia la aplicación interactiva ejecutando el siguiente comando:

```bash
streamlit run laboratorio_cnn.py
```

Se abrirá automáticamente una pestaña en tu navegador web (usualmente en `http://localhost:8501`) donde podrás manipular todos los parámetros del simulador en tiempo real.

---

## 🧠 ¿Por qué utilizamos CNN (Redes Neuronales Convolucionales)?

En la visión por computadora tradicional, tratar de entender una imagen conectando cada píxel directamente a una neurona tradicional (Redes Completamente Conectadas o *FCNs*) presenta graves limitaciones físicas e intelectuales. Las **CNN** resuelven esto gracias a tres principios fundamentales:

### 1. Invarianza Espacial (Localidad)
En una imagen, la información relevante es **local**. Un ojo, una arista o una textura está compuesta por píxeles que están pegados unos a otros. Las CNNs procesan la imagen utilizando filtros pequeños (ej. de 3x3) enfocados en estas vecindades locales, en lugar de intentar comprender toda la imagen gigante de un solo golpe.

### 2. Compartición de Parámetros (Eficiencia)
Si un filtro aprende a detectar un borde horizontal en la esquina superior izquierda de la imagen, ese mismo filtro es perfectamente capaz de detectar un borde horizontal en la esquina inferior derecha. Al deslizar el mismo filtro matemático (**Kernel**) sobre toda la imagen (convolución), reducimos drásticamente la cantidad de parámetros que la IA necesita memorizar.

### 3. Jerarquía de Características
Las CNNs imitan la corteza visual humana procesando la información de forma incremental:
*   **Capas iniciales:** Detectan rasgos ultra-simples como líneas, bordes, brillos y diagonales (lo que simula este laboratorio).
*   **Capas intermedias:** Combinan esas líneas para identificar formas geométricas, texturas, esquinas y siluetas.
*   **Capas profundas:** Agrupan las siluetas para reconocer conceptos complejos como rostros, gatos, automóviles u objetos de la vida real.

---

## 📂 Estructura de Archivos del Laboratorio Python

*   **`laboratorio_cnn.py`**: El código principal en Python que contiene el motor matemático de convolución, activación, submuestreo y la interfaz de usuario en Streamlit.
*   **`requirements.txt`**: Listado detallado de las bibliotecas y dependencias exactas para asegurar la portabilidad del entorno.
*   **`README.md`**: Esta guía didáctica y de instalación.
