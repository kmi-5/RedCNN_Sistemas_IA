import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from data import IMAGE_TEMPLATES, interpolate_16_to_64, KERNELS

# Compatibilidad de rerun para versiones antiguas de Streamlit
if not hasattr(st, "rerun"):
    st.rerun = st.experimental_rerun

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS
# ==========================================

st.set_page_config(
    page_title="Laboratorio CNN - Simulador de Visión Artificial",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo visual de alta fidelidad, con look cyberneuronal moderno, paleta de colores oscura, acentos neón y tarjetas con bordes redondeados.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Orbitron:wght@400;600;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Fondo general Cyberneuronal */
    .stApp {
        background: radial-gradient(circle at top, #0D1527 0%, #070B14 100%) !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Estructura general de la app */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1250px;
    }
    
    body {
        font-family: 'Inter', sans-serif;
        color: #E2E8F0;
        background-color: #070B14;
    }
    
    /* Cabecera de alta tecnología editorial */
    .editorial-title {
        font-family: 'Orbitron', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 50%, #9D4EDD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.01em;
        text-shadow: 0 0 30px rgba(0, 242, 254, 0.2);
    }
    
    .editorial-subtitle {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        font-size: 1.15rem;
        color: #94A3B8;
        line-height: 1.6;
        margin-bottom: 2.5rem;
    }
    
    .mono-badge {
        font-family: 'Fira Code', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #00F2FE;
        background-color: rgba(0, 242, 254, 0.08);
        padding: 6px 14px;
        border-radius: 9999px;
        display: inline-block;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(0, 242, 254, 0.3);
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.1);
    }
    
    /* Tarjetas de Diseño Premium Cyberneuronal */
    .premium-card {
        background-color: #0E1726;
        border: 1px solid rgba(0, 242, 254, 0.15);
        border-radius: 16px !important;
        padding: 1.8rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255,255,255,0.05);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .premium-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #00F2FE, #9D4EDD);
    }
    
    .premium-card h3 {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.35rem;
        color: #00F2FE;
        margin-top: 0;
        margin-bottom: 0.8rem;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.3);
    }
    
    .premium-card p {
        font-size: 0.95rem;
        color: #94A3B8;
        line-height: 1.6;
    }
    
    /* Cajas Didácticas (Callouts) */
    .story-callout {
        background-color: #111A2E;
        border: 1px solid rgba(157, 78, 221, 0.2);
        border-left: 5px solid #9D4EDD;
        padding: 1.5rem;
        border-radius: 12px;
        color: #E2E8F0;
        font-size: 0.95rem;
        line-height: 1.65;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .story-callout h4 {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.1rem;
        color: #9D4EDD;
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* Cajas de Alerta Cyber */
    .custom-alert {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-top: 1rem;
        font-size: 0.95rem;
        line-height: 1.5;
        border: 1px solid transparent;
    }
    .alert-success {
        background-color: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-left: 5px solid #10B981;
        color: #34D399;
    }
    .alert-info {
        background-color: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-left: 5px solid #3B82F6;
        color: #60A5FA;
    }
    .alert-warning {
        background-color: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-left: 5px solid #F59E0B;
        color: #FBBF24;
    }

    /* Matrices de pesos Estilizadas */
    .weight-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        max-width: 180px;
        margin: 1rem auto;
    }
    .weight-cell {
        font-family: 'Fira Code', monospace;
        font-size: 0.95rem;
        font-weight: 600;
        text-align: center;
        padding: 10px;
        background: #090E17;
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 8px;
        color: #00F2FE;
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.05);
    }
    
    /* Tablero de Juego */
    .game-container {
        background: linear-gradient(135deg, #111A2E 0%, #090D16 100%) !important;
        color: #E2E8F0 !important;
        border-radius: 16px !important;
        padding: 2.2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 242, 254, 0.08) !important;
        border: 1px solid rgba(0, 242, 254, 0.2) !important;
    }
    .game-container h4 {
        color: #00F2FE !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.6rem !important;
        margin-top: 0;
        margin-bottom: 0.8rem;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.2);
    }
    .game-container p {
        color: #94A3B8 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Sobrescribir estilos globales de Streamlit para botones y inputs */
    div.stButton > button {
        background-color: #121E36 !important;
        color: #00F2FE !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
        border-radius: 10px !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(0, 242, 254, 0.05) !important;
    }
    div.stButton > button:hover {
        background-color: #00F2FE !important;
        color: #070B14 !important;
        border-color: #00F2FE !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Estilos para pestañas de Streamlit */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(14, 23, 38, 0.5);
        padding: 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Orbitron', sans-serif;
        font-weight: 500;
        color: #94A3B8;
        border-radius: 8px;
        padding: 8px 16px;
        background-color: transparent;
        border: none;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00F2FE;
        background-color: rgba(0, 242, 254, 0.05);
    }
    .stTabs [aria-selected="true"] {
        color: #070B14 !important;
        background-color: #00F2FE !important;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 242, 254, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. FUNCIONES MATEMÁTICAS CLAVE DE CNN
# ==========================================

def convolve2d(input_matrix, kernel, stride=1, padding=0, bias=0):
    in_h, in_w = input_matrix.shape
    kh, kw = kernel.shape
    
    if padding > 0:
        padded = np.pad(input_matrix, padding, mode='constant', constant_values=0)
    else:
        padded = input_matrix.copy()
        
    pad_h, pad_w = padded.shape
    out_h = int((pad_h - kh) / stride) + 1
    out_w = int((pad_w - kw) / stride) + 1
    
    output = np.zeros((out_h, out_w), dtype=np.float32)
    for r in range(out_h):
        for c in range(out_w):
            start_r = r * stride
            start_c = c * stride
            patch = padded[start_r:start_r+kh, start_c:start_c+kw]
            output[r, c] = np.sum(patch * kernel) + bias
    return output

def relu_activation(matrix, threshold=0.0):
    return np.maximum(threshold, matrix)

def max_pooling2d(matrix, size=2, stride=2):
    in_h, in_w = matrix.shape
    out_h = int((in_h - size) / stride) + 1
    out_w = int((in_w - size) / stride) + 1
    
    output = np.zeros((out_h, out_w), dtype=np.float32)
    for r in range(out_h):
        for c in range(out_w):
            start_r = r * stride
            start_c = c * stride
            patch = matrix[start_r:start_r+size, start_c:start_c+size]
            output[r, c] = np.max(patch)
    return output

def avg_pooling2d(matrix, size=2, stride=2):
    in_h, in_w = matrix.shape
    out_h = int((in_h - size) / stride) + 1
    out_w = int((in_w - size) / stride) + 1
    
    output = np.zeros((out_h, out_w), dtype=np.float32)
    for r in range(out_h):
        for c in range(out_w):
            start_r = r * stride
            start_c = c * stride
            patch = matrix[start_r:start_r+size, start_c:start_c+size]
            output[r, c] = np.mean(patch)
    return output

def inject_noise(matrix, level):
    if level == 0:
        return matrix.copy()
    noise = np.random.normal(0, level, matrix.shape)
    return np.clip(matrix + noise, 0, 255)

def apply_blur(matrix, radius):
    if radius == 0:
        return matrix.copy()
    h, w = matrix.shape
    output = np.zeros_like(matrix)
    for r in range(h):
        for c in range(w):
            r_start = max(0, r - radius)
            r_end = min(h, r + radius + 1)
            c_start = max(0, c - radius)
            c_end = min(w, c + radius + 1)
            output[r, c] = np.mean(matrix[r_start:r_end, c_start:c_end])
    return output

def apply_occlusion(matrix, percent):
    if percent == 0:
        return matrix.copy()
    h, w = matrix.shape
    occlude_h = int(h * (percent / 100))
    occlude_w = int(w * (percent / 100))
    start_r = (h - occlude_h) // 2
    start_c = (w - occlude_w) // 2
    
    output = matrix.copy()
    output[start_r:start_r+occlude_h, start_c:start_c+occlude_w] = 15.0
    return output

def generate_grad_cam(matrix, focal_point, noise_lvl, blur_lvl, occlusion_lvl):
    h, w = matrix.shape
    y_idx, x_idx = np.indices((h, w))
    
    # Calcular centro focal base con perturbaciones aleatorias por distorsiones
    center_y = h * focal_point[0]
    center_x = w * focal_point[1]
    
    if occlusion_lvl > 0:
        center_y += np.random.uniform(-occlusion_lvl/10, occlusion_lvl/10)
        center_x += np.random.uniform(-occlusion_lvl/10, occlusion_lvl/10)
        
    dispersion = 1.0 + (blur_lvl * 0.15)
    radius = (min(h, w) * focal_point[2]) * dispersion
    
    dist_sq = (y_idx - center_y)**2 + (x_idx - center_x)**2
    intensity = np.maximum(0, 1.0 - (np.sqrt(dist_sq) / radius))
    
    if noise_lvl > 0:
        intensity += np.random.normal(0, noise_lvl / 150.0, intensity.shape)
        
    return np.clip(intensity, 0.0, 1.0)

def calculate_filter_reactivity(input_matrix, kernel):
    # Convolución discreta rápida con padding
    conv = convolve2d(input_matrix, kernel, stride=1, padding=1, bias=0)
    # Activación ReLU para retener activaciones positivas significativas
    activated = relu_activation(conv)
    # Retornar el promedio de activación como coeficiente de reactividad
    return float(np.mean(activated))

def estimate_training_metrics(config):
    # Generador de curvas realistas de entrenamiento
    lr = config["lr"]
    layers = config["layers"]
    filters = config["filters"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    kernel_sz = config["kernel_sz"]
    
    params = int(layers * (filters * filters * kernel_sz * kernel_sz) + 500 * filters)
    
    # Evaluar convergencia
    lr_dist = abs(np.log10(lr) - np.log10(0.001))
    lr_penalty = lr_dist * 0.35
    
    layer_mismatch = abs(layers - 3) * 0.08
    rate_of_convergence = max(0.02, 0.22 * (filters / 16.0) - lr_penalty - layer_mismatch)
    
    loss_history = []
    acc_history = []
    
    curr_loss = 2.45
    curr_acc = 10.0 + (filters / 2.0)
    max_train_score = min(99.2, 82.0 + (layers * 4.0) + (filters * 0.5))
    
    for epoch in range(1, 41):
        if epoch <= epochs:
            bounce = np.random.uniform(-0.02, 0.02)
            curr_loss = max(0.06, curr_loss - (curr_loss * rate_of_convergence) + bounce)
            curr_acc = min(max_train_score, curr_acc + (max_train_score - curr_acc) * (rate_of_convergence * 0.8) + bounce * 2.0)
        loss_history.append(round(curr_loss, 3))
        acc_history.append(round(curr_acc, 1))
        
    complexity = (layers * filters * kernel_sz * kernel_sz * epochs * 1000) / batch_size
    train_time = round(complexity / 850.0 + 0.5, 1)
    
    overfitting = "Saludable" if epochs <= 25 or filters < 24 else "Alto Riesgo (Sobreajuste Detectado)"
    
    return {
        "params": params,
        "loss_history": loss_history[:epochs],
        "acc_history": acc_history[:epochs],
        "time_seconds": train_time,
        "overfitting": overfitting,
        "final_loss": loss_history[epochs-1],
        "final_acc": acc_history[epochs-1]
    }


# ==========================================
# 3. INTERFAZ DE USUARIO PRINCIPAL
# ==========================================

# Cabecera editorial sofisticada
st.markdown('<div class="mono-badge">🧬 Vision por Computadora Artificial</div>', unsafe_allow_html=True)
st.markdown('<h1 class="editorial-title"> ¿Como procesan las redes CNN las imagenes? </h1>', unsafe_allow_html=True)
st.markdown('<div class="editorial-subtitle">Un simulador didáctico de vanguardia para experimentar, auditar y jugar en vivo con las Redes Neuronales Convolucionales (CNN) y entender cómo aprende a ver una máquina.</div>', unsafe_allow_html=True)

# ------------------------------------------
# BARRA LATERAL: CONTROL GLOBAL DE IMAGEN
# ------------------------------------------
st.sidebar.markdown("### ⚙️ Selector de Imagen Base")
selected_template_name = st.sidebar.selectbox(
    "1. Elige una plantilla:",
    list(IMAGE_TEMPLATES.keys()),
    key="sb_img"
)

template_data = IMAGE_TEMPLATES[selected_template_name]
base_low_res = np.array(template_data["low"], dtype=np.float32)

# Inicializar matriz de usuario persistente en session_state
if "user_matrix" not in st.session_state or "current_template_id" not in st.session_state or st.session_state["current_template_id"] != selected_template_name:
    st.session_state["user_matrix"] = base_low_res.copy()
    st.session_state["current_template_id"] = selected_template_name

user_raw_matrix = st.session_state["user_matrix"]
user_interpolated = interpolate_16_to_64(user_raw_matrix)

st.sidebar.info(f"🎯 **Propósito del Patrón:** {template_data['description']}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Hiperparámetros de Barrido")
stride_param = st.sidebar.slider("Paso (Stride):", min_value=1, max_value=3, value=1, help="Desplazamiento horizontal y vertical del filtro.", key="sb_stride")
padding_param = st.sidebar.slider("Relleno (Padding):", min_value=0, max_value=2, value=0, help="Bordes adicionales de píxeles rellenos con ceros.", key="sb_padding")
bias_param = st.sidebar.slider("Sesgo (Bias):", min_value=-150, max_value=150, value=0, step=10, help="Valor escalar sumado al resultado de la convolución.", key="sb_bias")

# ------------------------------------------
# CONFIGURACIÓN DE PESTAÑAS
# ------------------------------------------
tab_canvas, tab_story, tab_gallery, tab_microscope, tab_playground, tab_robustness, tab_game = st.tabs([
    "🎨 Lienzo Interactivo",
    "📖 Arquitectura Convolucional",
    "🖼️ Galería de Filtros",
    "🔎 Lupa y Microscopio",
    "🧠 Simulador de Entrenamiento",
    "🎯 Robustez y Grad-CAM",
    "🎮 Quizz"
])

# ==========================================
# PESTAÑA 1: LIENZO INTERACTIVO (DIBUJAR/EDITAR)
# ==========================================
with tab_canvas:
    st.markdown("### 🎨 Edita la Rejilla de Píxeles Crudos")
    st.markdown(
        "Esculpe tu propia imagen usando herramientas de brillo, filtros rápidos o un pincel de coordenadas. "
        "Observa en tiempo real cómo la interpolación matemática bilineal reconstruye la matriz en alta definición."
    )
    
    col_ctrl, col_visual = st.columns([1, 1.2])
    
    with col_ctrl:
        st.markdown("#### 🛠️ Herramientas de Brillo y Contraste")
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            if st.button("🔆 Aclarar Todo (+30)", use_container_width=True, help="Aumenta el brillo general de todos los píxeles.", key="btn_bright_cn"):
                st.session_state["user_matrix"] = np.clip(st.session_state["user_matrix"] + 30, 0, 255)
                st.rerun()
            if st.button("🔄 Invertir Colores", use_container_width=True, help="Invierte la escala de grises (negativo).", key="btn_inv_cn"):
                st.session_state["user_matrix"] = 255 - st.session_state["user_matrix"]
                st.rerun()
        with sub_col2:
            if st.button("🌙 Oscurecer Todo (-30)", use_container_width=True, help="Reduce el brillo general de todos los píxeles.", key="btn_dark_cn"):
                st.session_state["user_matrix"] = np.clip(st.session_state["user_matrix"] - 30, 0, 255)
                st.rerun()
            if st.button("🧹 Restaurar Plantilla", use_container_width=True, help="Restaura la imagen a su estado original de la plantilla.", key="btn_reset_cn"):
                st.session_state["user_matrix"] = base_low_res.copy()
                st.rerun()
                
        st.markdown("---")
        st.markdown("#### 🖌️ Pincel de Precisión de Píxeles")
        st.markdown("Selecciona coordenadas exactas y define su intensidad de brillo:")
        
        brush_r = st.slider("Fila (Y - Vertical):", 0, 15, 8, key="brush_r_slider_cn")
        brush_c = st.slider("Columna (X - Horizontal):", 0, 15, 8, key="brush_c_slider_cn")
        brush_val = st.slider("Brillo del Pincel (0 = Negro, 255 = Blanco):", 0, 255, 128, key="brush_val_slider_cn")
        
        if st.button("✍️ Pintar Píxel Seleccionado", use_container_width=True, type="primary", key="btn_paint_cn"):
            st.session_state["user_matrix"][brush_r, brush_c] = brush_val
            st.toast(f"¡Píxel en (Fila {brush_r}, Columna {brush_c}) pintado con brillo {brush_val}!", icon="✍️")
            st.rerun()
            
    with col_visual:
        st.markdown("##### 📐 Entrada Rejilla Base Cruda (16x16)")
        fig_raw = px.imshow(
            user_raw_matrix,
            color_continuous_scale="gray",
            zmin=0, zmax=255
        )
        fig_raw.update_layout(margin=dict(l=5, r=5, t=5, b=5), height=240, coloraxis_showscale=False)
        st.plotly_chart(fig_raw, use_container_width=True)
        
        st.markdown("##### 🔬 Reconstrucción por Interpolación (64x64)")
        fig_inter = px.imshow(
            user_interpolated,
            color_continuous_scale="gray",
            zmin=0, zmax=255
        )
        fig_inter.update_layout(margin=dict(l=5, r=5, t=5, b=5), height=240, coloraxis_showscale=False)
        st.plotly_chart(fig_inter, use_container_width=True)

    # 📡 MONITOR DE RESONANCIA NEURONAL EN TIEMPO REAL (CAPA 1 DE LA CNN)
    st.markdown("---")
    st.markdown("""
    <div class="premium-card">
        <h3>📡 Monitor de Resonancia de Características en Tiempo Real (Capa 1 de la CNN)</h3>
        <p>A medida que esculpes y modificas tu imagen pixel por pixel, las neuronas artificiales de la primera capa convolucional de la CNN reaccionan selectivamente en microsegundos. Este osciloscopio en vivo calcula qué tan fuerte se activan los detectores de patrones geométricos de bajo nivel:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calcular activaciones usando los kernels definidos en KERNELS
    v_act = calculate_filter_reactivity(user_raw_matrix, KERNELS["Bordes Verticales"]["matrix"])
    h_act = calculate_filter_reactivity(user_raw_matrix, KERNELS["Bordes Horizontales"]["matrix"])
    l_act = calculate_filter_reactivity(user_raw_matrix, KERNELS["Laplaciano (Bordes general)"]["matrix"])
    e_act = calculate_filter_reactivity(user_raw_matrix, KERNELS["Relieve (Emboss)"]["matrix"])
    
    # Normalización dinámica para obtener porcentajes dinámicos espectaculares (0 a 100%)
    max_act = max(v_act, h_act, l_act, e_act, 1e-5)
    v_pct = round((v_act / max_act) * 100.0, 1)
    h_pct = round((h_act / max_act) * 100.0, 1)
    l_pct = round((l_act / max_act) * 100.0, 1)
    e_pct = round((e_act / max_act) * 100.0, 1)
    
    col_pct1, col_pct2 = st.columns([1.2, 2.5])
    
    with col_pct1:
        st.markdown("#### ⚙️ Diagnóstico de Activación")
        # Mostrar explicaciones rápidas en base al filtro que tiene más activación
        max_f = max(v_pct, h_pct, l_pct, e_pct)
        if max_f == v_pct:
            st.markdown("""
            <div class="custom-alert alert-info">
                <strong>💡 Patrón Dominante: Bordes Verticales</strong><br>
                Tu diseño posee un marcado contraste horizontal, estimulando fuertemente las neuronas que diferencian perfiles laterales.
            </div>
            """, unsafe_allow_html=True)
        elif max_f == h_pct:
            st.markdown("""
            <div class="custom-alert alert-info">
                <strong>💡 Patrón Dominante: Bordes Horizontales</strong><br>
                Tus trazos poseen una gran cantidad de líneas horizontales paralelas, estimulando neuronas de barrido vertical.
            </div>
            """, unsafe_allow_html=True)
        elif max_f == l_pct:
            st.markdown("""
            <div class="custom-alert alert-warning">
                <strong>💡 Patrón Dominante: Laplaciano (Esquinas/Ruido)</strong><br>
                Hay transiciones multidireccionales abruptas o ruido de alta frecuencia, estimulando detectores de contraste omnidireccionales.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="custom-alert alert-success">
                <strong>💡 Patrón Dominante: Relieve (Texturas)</strong><br>
                La imagen tiene sombreados tridimensionales o gradientes continuos, activando filtros de profundidad estructural.
            </div>
            """, unsafe_allow_html=True)
            
    with col_pct2:
        # Gráfico horizontal moderno de Plotly con colores estilo cyberneuronal neón
        fig_react = go.Figure(go.Bar(
            x=[v_pct, h_pct, l_pct, e_pct],
            y=["Bordes Verticales", "Bordes Horizontales", "Esquinas / Ruido", "Textura / Relieve"],
            orientation='h',
            text=[f"{v_pct}%", f"{h_pct}%", f"{l_pct}%", f"{e_pct}%"],
            textposition='auto',
            marker=dict(
                color=[v_pct, h_pct, l_pct, e_pct],
                colorscale=[[0.0, '#9D4EDD'], [0.5, '#4FACFE'], [1.0, '#00F2FE']],
                line=dict(color='#00F2FE', width=1.5)
            )
        ))
        
        fig_react.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title=dict(
                    text="Magnitud Relativa de Resonancia (%)",
                    font=dict(color='#94A3B8', size=11, family='Orbitron')
                ),
                range=[0, 105], 
                gridcolor='rgba(255,255,255,0.05)',
                tickfont=dict(color='#94A3B8')
            ),
            yaxis=dict(
                tickfont=dict(color='#E2E8F0', size=12, family='Orbitron'),
                gridcolor='rgba(255,255,255,0.05)'
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            height=200,
            showlegend=False
        )
        st.plotly_chart(fig_react, use_container_width=True)

# ==========================================
# PESTAÑA 2: EL VIAJE DEL PÍXEL (STORYTELLING DETALLADO)
# ==========================================
with tab_story:
    st.markdown("## 📖 Las Etapas Matemáticas del Sensor")
    st.markdown("Acompaña a tu imagen personalizada en su recorrido a través de la arquitectura convolucional:")
    
    sub_tab_conv, sub_tab_relu, sub_tab_pool = st.tabs([
        "1. Convolución y Filtros",
        "2. Rectificación (ReLU)",
        "3. Compresión (Pooling)"
    ])
    
    with sub_tab_conv:
        col_t, col_v = st.columns([1, 1.2])
        with col_t:
            st.markdown("### Filtros Convolucionales")
            st.markdown(
                "La red neuronal no analiza la imagen completa de golpe, sino que desliza pequeños "
                "sensores llamados **kernels o filtros de 3x3**. Al colocarse sobre un parche, "
                "calculan el producto escalar local."
            )
            
            selected_kernel_name = st.selectbox(
                "Elige un filtro de barrido:",
                list(KERNELS.keys()),
                index=2,
                key="story_kernel"
            )
            
            active_kernel = KERNELS[selected_kernel_name]["matrix"]
            st.markdown(f"**Analogía Teórica:** *{KERNELS[selected_kernel_name]['analogy']}*")
            st.markdown("**Pesos de la Matriz:**")
            st.markdown(f"""
            <div class="weight-grid">
                <div class="weight-cell">{active_kernel[0,0]:.1f}</div>
                <div class="weight-cell">{active_kernel[0,1]:.1f}</div>
                <div class="weight-cell">{active_kernel[0,2]:.1f}</div>
                <div class="weight-cell">{active_kernel[1,0]:.1f}</div>
                <div class="weight-cell">{active_kernel[1,1]:.1f}</div>
                <div class="weight-cell">{active_kernel[1,2]:.1f}</div>
                <div class="weight-cell">{active_kernel[2,0]:.1f}</div>
                <div class="weight-cell">{active_kernel[2,1]:.1f}</div>
                <div class="weight-cell">{active_kernel[2,2]:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_v:
            st.markdown("##### 🔬 Salida de la Convolución (Feature Map)")
            conv_res = convolve2d(user_interpolated, active_kernel, stride=stride_param, padding=padding_param, bias=bias_param)
            
            fig_conv = px.imshow(conv_res, color_continuous_scale="electric")
            fig_conv.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380)
            st.plotly_chart(fig_conv, use_container_width=True)
            
    with sub_tab_relu:
        col_t, col_v = st.columns([1, 1.2])
        with col_t:
            st.markdown("### El Umbral de Activación: ReLU")
            st.markdown(
                "La convolución es estrictamente lineal. Para modelar la complejidad curvada del mundo real, "
                "aplicamos la función de activación **ReLU (Rectified Linear Unit)**, que trunca instantáneamente "
                "los valores negativos a cero perfecto (`0`), apagando las neuronas inactivas."
            )
            
            threshold_slider = st.slider("Umbral de Truncamiento (ReLU):", min_value=-50.0, max_value=50.0, value=0.0, step=5.0, key="story_relu")
            st.latex(r"f(x) = \max(\theta, x)")
            
        with col_v:
            st.markdown("##### 💡 Salida Rectificada de la Neurona")
            conv_for_relu = convolve2d(user_interpolated, KERNELS["Bordes Verticales"]["matrix"], stride=1, padding=0)
            relu_res = relu_activation(conv_for_relu, threshold=threshold_slider)
            
            fig_relu = px.imshow(relu_res, color_continuous_scale="inferno")
            fig_relu.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380)
            st.plotly_chart(fig_relu, use_container_width=True)
            
    with sub_tab_pool:
        col_t, col_v = st.columns([1, 1.2])
        with col_t:
            st.markdown("### Compresión Espacial: Pooling")
            st.markdown(
                "Para lograr **invarianza espacial** ante ligeros desplazamientos y disminuir la carga "
                "computacional, reducimos la resolución mediante submuestreos. "
                "Max-Pooling retiene el rasgo de contraste más fuerte en cada ventana local."
            )
            
            pool_method = st.radio("Método de submuestreo:", ["Max Pooling", "Average Pooling"], key="story_pool_method")
            pool_sz = st.slider("Tamaño de ventana local:", min_value=2, max_value=4, value=2, key="story_pool_sz")
            
        with col_v:
            st.markdown("##### 🔬 Mapa Comprimido resultante")
            conv_for_pool = convolve2d(user_interpolated, KERNELS["Bordes Verticales"]["matrix"], stride=1, padding=0)
            if pool_method == "Max Pooling":
                pool_res = max_pooling2d(conv_for_pool, size=pool_sz, stride=pool_sz)
            else:
                pool_res = avg_pooling2d(conv_for_pool, size=pool_sz, stride=pool_sz)
                
            fig_pool = px.imshow(pool_res, color_continuous_scale="plasma")
            fig_pool.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350)
            st.plotly_chart(fig_pool, use_container_width=True)
            
            reduction_pct = 100 - (pool_res.size / conv_for_pool.size * 100)
            st.success(f"¡Has logrado comprimir un **{reduction_pct:.1f}%** de los datos originales sin perder siluetas clave!")

# ==========================================
# PESTAÑA 3: CALEIDOSCOPIO DE FEATURE MAPS
# ==========================================
with tab_gallery:
    st.markdown("### 🖼️ Caleidoscopio de Canales de Abstracción")
    st.markdown("Evalúa múltiples representaciones al mismo tiempo. Activa y desactiva filtros de barrido:")
    
    active_filters = st.multiselect(
        "Filtros convolucionales activos:",
        options=list(KERNELS.keys()),
        default=["Identidad", "Bordes Verticales", "Bordes Horizontales", "Laplaciano (Bordes general)", "Relieve (Emboss)"],
        key="gallery_filters"
    )
    
    if active_filters:
        grid_cols = st.columns(len(active_filters))
        for idx, k_name in enumerate(active_filters):
            with grid_cols[idx]:
                k_data = KERNELS[k_name]
                out_m = convolve2d(user_interpolated, k_data["matrix"], stride=stride_param, padding=padding_param, bias=bias_param)
                
                st.markdown(f"**{k_name}**")
                fig_cell = px.imshow(out_m, color_continuous_scale="electric")
                fig_cell.update_layout(margin=dict(l=5, r=5, t=5, b=5), height=220, coloraxis_showscale=False)
                st.plotly_chart(fig_cell, use_container_width=True)
                st.caption(f"<small>*{k_data['description']}*</small>", unsafe_allow_html=True)
    else:
        st.warning("Elige al menos un filtro de la lista superior para renderizar la galería.")
 
# ==========================================
# PESTAÑA 4: LUPA Y MICROSCOPIO MATEMÁTICO
# ==========================================
with tab_microscope:
    st.markdown("### ⚖️ Lupa Sincronizada de Coordenadas")
    st.markdown("Selecciona una posición de la rejilla para desglosar paso a paso el cálculo de la convolución:")
    
    comp_c1, comp_c2 = st.columns(2)
    with comp_c1:
        f_left = st.selectbox("Filtro Izquierdo:", list(KERNELS.keys()), index=2, key="micro_left_f")
    with comp_c2:
        f_right = st.selectbox("Filtro Derecho:", list(KERNELS.keys()), index=6, key="micro_right_f")
        
    out_left = convolve2d(user_interpolated, KERNELS[f_left]["matrix"], stride=1, padding=0)
    out_right = convolve2d(user_interpolated, KERNELS[f_right]["matrix"], stride=1, padding=0)
    
    render_c1, render_c2 = st.columns(2)
    with render_c1:
        fig_l = px.imshow(out_left, color_continuous_scale="electric")
        fig_l.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280)
        st.plotly_chart(fig_l, use_container_width=True)
    with render_c2:
        fig_r = px.imshow(out_right, color_continuous_scale="plasma")
        fig_r.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280)
        st.plotly_chart(fig_r, use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### 🔬 Lupa Matemática de Vecindario")
    
    lc1, lc2 = st.columns(2)
    px_x = lc1.slider("Posición Píxel X:", 0, 63, 32, key="micro_x_p")
    px_y = lc2.slider("Posición Píxel Y:", 0, 63, 32, key="micro_y_p")
    
    try:
        val_base = user_interpolated[px_y, px_x]
        val_l = out_left[px_y, px_x]
        val_r = out_right[px_y, px_x]
        
        st.markdown(f"""
        <div class="premium-card">
            <h3>Desglose de Lectura en Coordenada [{px_y}, {px_x}]</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; text-align: center;">
                <div>
                    <span style="font-size: 11px; color: #718096; text-transform: uppercase; font-weight: bold;">Intensidad Base</span>
                    <h2 style="font-family: monospace; color: #0F172A; margin: 5px 0;">{int(val_base)}</h2>
                </div>
                <div>
                    <span style="font-size: 11px; color: #3B82F6; text-transform: uppercase; font-weight: bold;">{f_left}</span>
                    <h2 style="font-family: monospace; color: #3B82F6; margin: 5px 0;">{val_l:.2f}</h2>
                </div>
                <div>
                    <span style="font-size: 11px; color: #10B981; text-transform: uppercase; font-weight: bold;">{f_right}</span>
                    <h2 style="font-family: monospace; color: #10B981; margin: 5px 0;">{val_r:.2f}</h2>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Fórmula explicada en vivo
        st.markdown("##### 🧮 Operación de Barrido Acumulativa:")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"**1. Vecindario 3x3 de la Imagen alrededor del píxel:**")
            padded_grid = np.pad(user_interpolated, 1, mode="constant", constant_values=0)
            neighborhood = padded_grid[px_y:px_y+3, px_x:px_x+3]
            st.dataframe(pd.DataFrame(neighborhood.astype(int)), use_container_width=True)
            
        with col_m2:
            st.markdown(f"**2. Multiplicación paso a paso de pesos para '{f_left}':**")
            k_weights = KERNELS[f_left]["matrix"]
            st.dataframe(pd.DataFrame(k_weights), use_container_width=True)
            
            # Suma acumulativa
            formulas = []
            for r in range(3):
                for c in range(3):
                    formulas.append(f"({int(neighborhood[r,c])} × {k_weights[r,c]:.1f})")
            st.markdown("**Cálculo en vivo:**")
            st.code(" + \n".join(formulas) + f"\n\n+ Sesgo(Bias)={bias_param}\n= {val_l:.2f}", language="text")
            
    except Exception as e:
        st.error(f"Error al calcular la lupa matemática: {e}")

# ==========================================
# PESTAÑA 5: SIMULADOR DE ENTRENAMIENTO (DEEP PLAYGROUND)
# ==========================================
with tab_playground:
    st.markdown("### 🧠 Simulador del Comportamiento del Aprendizaje Profundo")
    st.markdown(
        "Las redes neuronales aprenden ajustando iterativamente sus parámetros de forma automática. "
        "Ajusta los hiperparámetros para modelar la velocidad de convergencia y las curvas de precisión:"
    )
    
    col_ctrl, col_charts = st.columns([1, 1.2])
    
    with col_ctrl:
        st.markdown("##### 🧱 Configuración del Modelo")
        filters_val = st.slider("Cantidad de Filtros:", 8, 64, 16, step=8, key="play_filters")
        layers_val = st.slider("Capas Convolucionales:", 1, 5, 3, key="play_layers")
        kernel_size_val = st.slider("Tamaño de Kernel:", 3, 7, 3, step=2, key="play_kernel_sz")
        
        st.markdown("##### 🎛️ Configuración de Entrenamiento")
        lr_val = st.select_slider("Tasa de Aprendizaje (Learning Rate):", [0.1, 0.01, 0.001, 0.0001], value=0.001, key="play_lr")
        epochs_val = st.slider("Épocas de Entrenamiento (Epochs):", 5, 40, 15, key="play_epochs")
        batch_size_val = st.select_slider("Tamaño de Lote (Batch Size):", [8, 16, 32, 64], value=32, key="play_batch_sz")
        
        # Calcular estimaciones en vivo
        res_metrics = estimate_training_metrics({
            "filters": filters_val,
            "layers": layers_val,
            "kernel_sz": kernel_size_val,
            "lr": lr_val,
            "epochs": epochs_val,
            "batch_size": batch_size_val
        })
        
    with col_charts:
        st.markdown("##### 📈 Curvas de Rendimiento Estimadas")
        
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=res_metrics["loss_history"], name="Pérdida (Loss)", line=dict(color="#EF4444", width=3)))
        fig_loss.update_layout(title="Curva de Pérdida (Loss Decay)", height=200, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_loss, use_container_width=True)
        
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(y=res_metrics["acc_history"], name="Precisión (Accuracy)", line=dict(color="#10B981", width=3)))
        fig_acc.update_layout(title="Curva de Precisión (Accuracy %)", height=200, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_acc, use_container_width=True)
        
        # Estadísticas resumidas
        col_st1, col_st2, col_st3 = st.columns(3)
        col_st1.metric("Parámetros Totales", f"{res_metrics['params']:,}")
        col_st2.metric("Precisión Alcanzada", f"{res_metrics['final_acc']:.1f}%")
        col_st3.metric("Tiempo Estimado", f"{res_metrics['time_seconds']}s")

# ==========================================
# PESTAÑA 6: AUDITOR DE ROBUSTEZ Y GRAD-CAM
# ==========================================
with tab_robustness:
    st.markdown("### 🎯 Robustez y Explicabilidad (Grad-CAM)")
    st.markdown(
        "¿Qué tan bien resiste el modelo ante distorsiones físicas? "
        "Ajusta los sliders para inyectar ruido o desviar el foco visual del modelo, "
        "y observa cómo se desplaza el mapa de calor de atención **Grad-CAM**:"
    )
    
    col_sliders, col_views = st.columns([1, 1.3])
    
    with col_sliders:
        st.markdown("##### 🌪️ Inyectar Perturbaciones")
        noise_level = st.slider("Ruido Gaussiano:", 0.0, 50.0, 0.0, step=5.0, key="robust_noise")
        blur_radius = st.slider("Radio de Desenfoque (Blur):", 0, 5, 0, key="robust_blur")
        occlusion_percent = st.slider("Bloqueo Central (Occlude %):", 0, 60, 0, key="robust_occlude")
        
        st.markdown("""
        ---
        💡 **Concepto de Explicabilidad:**
        Grad-CAM genera un mapa de calor que resalta las zonas de la imagen que tuvieron mayor peso e impacto 
        en la decisión final de la IA. Si la imagen es muy ruidosa u ocluida, el foco de la IA se desvía 
        o se expande perdiendo foco estructural.
        """)
        
    with col_views:
        # Aplicar distorsiones a la imagen
        noisy_img = inject_noise(user_interpolated, noise_level)
        blurred_img = apply_blur(noisy_img, blur_radius)
        degraded_img = apply_occlusion(blurred_img, occlusion_percent)
        
        # Calcular mapa Grad-CAM
        focal_p = template_data.get("focalPoint", [0.5, 0.5, 0.22])
        if len(focal_p) == 2:
            # Añadir radio por defecto si no lo tiene
            focal_p = [focal_p[0], focal_p[1], 0.22]
            
        grad_cam_map = generate_grad_cam(degraded_img, focal_p, noise_level, blur_radius, occlusion_percent)
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("**Imagen Distorsionada (Entrada):**")
            fig_v1 = px.imshow(degraded_img, color_continuous_scale="gray")
            fig_v1.update_layout(margin=dict(l=5, r=5, t=5, b=5), height=260, coloraxis_showscale=False)
            st.plotly_chart(fig_v1, use_container_width=True)
            
        with col_v2:
            st.markdown("**Atención del Modelo (Grad-CAM):**")
            fig_v2 = px.imshow(grad_cam_map, color_continuous_scale="turbo")
            fig_v2.update_layout(margin=dict(l=5, r=5, t=5, b=5), height=260, coloraxis_showscale=False)
            st.plotly_chart(fig_v2, use_container_width=True)

# ==========================================
# PESTAÑA 7: TRIVIA DE INTEGRACIÓN (GAME)
# ==========================================
with tab_game:
    st.markdown("""
    <div class="game-container" style="background-color: #0E1726; padding: 20px; border-radius: 16px; border: 1px solid rgba(0, 242, 254, 0.2); margin-bottom: 20px;">
        <h4 style="margin-top: 0; color: #00F2FE;">🧠 Trivia y Desafío de Comprensión CNN</h4>
        <p style="color: #94A3B8; margin-bottom: 0;">¡Exprime tus conocimientos! Este juego interactivo te permitirá integrar los conceptos clave que has explorado en este laboratorio: convoluciones, activaciones, pooling, explicabilidad y robustez.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Lista de preguntas del Quiz
    quiz_questions = [
        {
            "question": "1. Cuando aplicamos un filtro con pesos opuestos en las columnas laterales (como un Kernel de Sobel Vertical), ¿qué tipo de patrón resalta con mayor intensidad?",
            "options": [
                "Las zonas de color uniforme y texturas lisas.",
                "Las líneas y transiciones verticales abruptas de contraste de izquierda a derecha.",
                "El promedio general de brillo, suavizando todos los ruidos locales.",
                "Toda la imagen con un brillo duplicado y plano."
            ],
            "correct": 1,
            "explanation": "¡Excelente! Las columnas con pesos opuestos calculan la diferencia (gradiente) de intensidad entre el lado izquierdo y el derecho, revelando bordes verticales nítidos."
        },
        {
            "question": "2. ¿Por qué es fundamental la función de activación ReLU (max(0, x)) después de una convolución?",
            "options": [
                "Para invertir todos los colores de la imagen haciéndola más artística.",
                "Para comprimir la imagen reduciendo su ancho y alto a la mitad.",
                "Para introducir no linealidad, descartando respuestas negativas opuestas al filtro y permitiendo a la red aprender patrones geométricos complejos.",
                "Para rellenar las esquinas de la imagen con píxeles blancos artificiales."
            ],
            "correct": 2,
            "explanation": "¡Brillante! Sin ReLU u otras no linealidades, la red completa se reduciría algebraicamente a una sola operación lineal plana, imposibilitando el reconocimiento de curvas y formas complejas."
        },
        {
            "question": "3. ¿Qué beneficio clave aporta la operación de 'Max Pooling' (submuestreo) al flujo de la red convolucional?",
            "options": [
                "Duplica el número de parámetros reduciendo la velocidad de entrenamiento.",
                "Aporta invariancia por traslación espacial y comprime la matriz descartando información secundaria para retener sólo la respuesta más fuerte de cada cuadrante.",
                "Elimina por completo los canales RGB convirtiendo todo a escala de grises.",
                "Aplana por completo todos los gradientes de retropropagación para que los bordes dejen de ser visibles."
            ],
            "correct": 1,
            "explanation": "¡Correcto! Max Pooling selecciona el valor máximo en vecindades (por ejemplo, de 2x2), logrando que la red detecte la característica sin importar si está ligeramente desplazada de posición."
        },
        {
            "question": "4. Si el mapa de calor de Grad-CAM de un clasificador de 'Gatos' ilumina intensamente el pasto de fondo en vez del cuerpo del gato, ¿qué nos indica esto científicamente?",
            "options": [
                "Que el gato está perfectamente mimetizado en el pasto y la red es súper precisa.",
                "Que el modelo sufre de un sesgo contextual perjudicial (adivinando por el fondo verde) en lugar de aprender los rasgos anatómicos reales del gato.",
                "Que la función de coste ya converge a cero de forma ideal y no hay nada que auditar.",
                "Que la tasa de aprendizaje es demasiado baja."
            ],
            "correct": 1,
            "explanation": "¡Exacto! Los mapas de explicabilidad como Grad-CAM nos permiten desenmascarar correlaciones falsas: cuando la IA 'adivina' una clase observando elementos incidentales del entorno."
        },
        {
            "question": "5. ¿De qué forma podemos proteger de antemano a una CNN para que no falle ante el ruido, niebla, oclusión o desenfoques en el mundo real?",
            "options": [
                "Entrenando la red únicamente con imágenes puras y perfectas de laboratorio.",
                "Aplicando un stride de valor 10 en todas las capas convolucionales.",
                "Implementando Aumentación de Datos (Data Augmentation), inyectando ruido gaussiano, desenfoques y recortes de manera controlada durante el entrenamiento.",
                "Desactivando la capa de pooling para conservar la nitidez de la imagen."
            ],
            "correct": 2,
            "explanation": "¡Espectacular! Al forzar al modelo a procesar muestras perturbadas artificialmente durante la fase de optimización, los filtros aprenden a retener abstracciones estructurales robustas e inmunes al desorden cotidiano."
        }
    ]
    
    # Inicializar estado del Quiz si no existe
    if "st_quiz_index" not in st.session_state:
        st.session_state.st_quiz_index = 0
    if "st_quiz_score" not in st.session_state:
        st.session_state.st_quiz_score = 0
    if "st_quiz_submitted" not in st.session_state:
        st.session_state.st_quiz_submitted = False
    if "st_quiz_selected_opt" not in st.session_state:
        st.session_state.st_quiz_selected_opt = None
    if "st_quiz_finished" not in st.session_state:
        st.session_state.st_quiz_finished = False

    # Si ya se terminó el quiz, mostrar resumen final
    if st.session_state.st_quiz_finished:
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background-color: #0E1726; border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 16px; box-shadow: 0 0 15px rgba(0,242,254,0.1);">
            <h2 style="margin-top: 0; color: #00F2FE; font-family: 'Orbitron', sans-serif;">🎉 ¡Quiz Completado!</h2>
            <p style="font-size: 1.2rem; color: #E2E8F0;">Has respondido todas las preguntas de evaluación.</p>
            <hr style="border: 0; border-top: 1px solid rgba(0, 242, 254, 0.2); margin: 20px 0;" />
        </div>
        """, unsafe_allow_html=True)
        
        col_score1, col_score2 = st.columns(2)
        score_percentage = (st.session_state.st_quiz_score / len(quiz_questions)) * 100
        col_score1.metric("Puntuación Final", f"{st.session_state.st_quiz_score} / {len(quiz_questions)}", f"{score_percentage:.1f}%")
        
        if score_percentage >= 80.0:
            st.success("🏆 ¡Felicitaciones! Demuestras un entendimiento sobresaliente sobre las arquitecturas de visión convolucionales y explicabilidad.")
        elif score_percentage >= 50.0:
            st.info("⚡ ¡Buen trabajo! Tienes bases sólidas de CNN, aunque te vendría bien repasar algunos conceptos de pooling o Grad-CAM para perfeccionar.")
        else:
            st.warning("⏳ Sigue explorando el laboratorio: repasa las pestañas de 'El Viaje del Píxel' y 'Lupa y Microscopio' para fortalecer los fundamentos.")
            
        if st.button("🔄 Volver a jugar", key="quiz_restart_btn_final"):
            st.session_state.st_quiz_index = 0
            st.session_state.st_quiz_score = 0
            st.session_state.st_quiz_submitted = False
            st.session_state.st_quiz_selected_opt = None
            st.session_state.st_quiz_finished = False
            st.rerun()
            
    else:
        current_q_idx = st.session_state.st_quiz_index
        q_data = quiz_questions[current_q_idx]
        
        st.markdown(f"##### Pregunta {current_q_idx + 1} de {len(quiz_questions)}")
        st.markdown(f"**{q_data['question']}**")
        
        # Opciones
        selected_option = st.radio(
            "Selecciona tu respuesta:",
            q_data["options"],
            index=st.session_state.st_quiz_selected_opt if st.session_state.st_quiz_selected_opt is not None else 0,
            key=f"q_radio_{current_q_idx}",
            disabled=st.session_state.st_quiz_submitted
        )
        
        # Guardar opción seleccionada en el estado
        opt_index = q_data["options"].index(selected_option)
        st.session_state.st_quiz_selected_opt = opt_index
        
        col_btn1, col_btn2 = st.columns([1, 1])
        
        if not st.session_state.st_quiz_submitted:
            if col_btn1.button("✔️ Enviar Respuesta", key="quiz_submit_btn_run", use_container_width=True):
                st.session_state.st_quiz_submitted = True
                if opt_index == q_data["correct"]:
                    st.session_state.st_quiz_score += 1
                st.rerun()
        else:
            is_correct = opt_index == q_data["correct"]
            if is_correct:
                st.success("🎉 ¡Correcto!")
            else:
                correct_text = q_data["options"][q_data["correct"]]
                st.error(f"❌ Incorrecto. La opción correcta es: **{correct_text}**")
                
            st.info(f"💡 **Explicación:** {q_data['explanation']}")
            
            is_last_q = (current_q_idx == len(quiz_questions) - 1)
            btn_text = "Ver resultados 🏁" if is_last_q else "Siguiente pregunta ➡️"
            
            if col_btn2.button(btn_text, key="quiz_next_btn_run", use_container_width=True):
                if is_last_q:
                    st.session_state.st_quiz_finished = True
                    if st.session_state.st_quiz_score == len(quiz_questions):
                        st.balloons()
                else:
                    st.session_state.st_quiz_index += 1
                st.session_state.st_quiz_submitted = False
                st.session_state.st_quiz_selected_opt = None
                st.rerun()
