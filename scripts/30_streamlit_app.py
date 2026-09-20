"""
CPAK - Streamlit Web UI
Análisis clínico de alineación femoral (LDFA, MPTA, aHKA, JLO).

Launch:
    conda run -n physis_seg streamlit run scripts/30_streamlit_app.py
"""
import importlib.util
import json
import os
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Ultralytics escribe su config en el HOME; en contenedores eso puede fallar.
# Se apunta a un directorio temporal escribible (no-op molesto en local).
os.environ.setdefault(
    "YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "Ultralytics")
)

# Compat shim: streamlit-drawable-canvas espera streamlit.elements.image.image_to_url
# (API antigua). En Streamlit 1.57 se movio a image_utils con otra firma.
try:
    import streamlit.elements.image as st_image_mod
    if not hasattr(st_image_mod, "image_to_url"):
        from streamlit.elements.lib.image_utils import image_to_url as _new_image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig

        def _compat_image_to_url(image, width, clamp, channels, output_format, image_id):
            return _new_image_to_url(
                image=image,
                layout_config=LayoutConfig(width=width),
                clamp=clamp,
                channels=channels,
                output_format=output_format,
                image_id=image_id,
            )

        st_image_mod.image_to_url = _compat_image_to_url
except Exception:
    pass

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Pegado desde portapapeles (Ctrl+V)
# Streamlit no soporta pegar imágenes en st.file_uploader. Este handler (en un
# iframe del mismo origen) escucha el evento 'paste' en el documento padre y
# inyecta la imagen pegada como File en el <input type="file"> del uploader,
# de modo que el resto de la app funciona sin cambios.
# ---------------------------------------------------------------------------
PASTE_HANDLER_HTML = """
<script>
(function () {
  let P, D;
  try {
    P = window.parent;
    D = P.document;
    if (!D || !D.body) return;
  } catch (e) {
    console.warn('[CPAK] Paste no disponible (iframe sin same-origin).');
    return;
  }

  function inject(blob, name) {
    const inp = D.querySelector('input[type="file"]');
    if (!inp) return false;
    try {
      const file = new P.File([blob], name, { type: blob.type || 'image/png' });
      const dt = new P.DataTransfer();
      dt.items.add(file);
      inp.files = dt.files;
      inp.dispatchEvent(new P.Event('input', { bubbles: true }));
      inp.dispatchEvent(new P.Event('change', { bubbles: true }));
      return true;
    } catch (e) {
      console.error('[CPAK] No se pudo inyectar la imagen pegada', e);
      return false;
    }
  }

  function onPaste(e) {
    const items = (e.clipboardData || {}).items || [];
    for (let i = 0; i < items.length; i++) {
      if (items[i].type && items[i].type.indexOf('image') === 0) {
        const blob = items[i].getAsFile();
        if (blob) {
          const ext = (blob.type.split('/')[1] || 'png').replace('jpeg', 'jpg');
          const ok = inject(blob, 'pasted_' + Date.now() + '.' + ext);
          if (ok) e.preventDefault();
        }
        return;
      }
    }
  }

  // Evita acumular listeners en cada re-render de Streamlit
  if (P.__cpakPasteHandler) {
    D.removeEventListener('paste', P.__cpakPasteHandler);
  }
  P.__cpakPasteHandler = onPaste;
  D.addEventListener('paste', onPaste);
})();
</script>
"""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
PIPELINE_PATH = BASE_DIR / "scripts" / "13_final_inference.py"

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CPAK · Inferencia Clínica",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS tema profesional
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* Fondo principal más oscuro */
[data-testid="stAppViewContainer"] {
    background-color: #0f1117;
}
[data-testid="stSidebar"] {
    background-color: #161b27;
}

/* Métricas más grandes */
[data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    color: #8b9ab1 !important;
}

/* Título principal */
.cpak-title {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(90deg, #4f9cf9, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.cpak-subtitle {
    font-size: 14px;
    color: #8b9ab1;
    margin-bottom: 24px;
}

/* Tarjeta de clasificación CPAK */
.cpak-badge {
    padding: 10px 18px;
    border-radius: 8px;
    text-align: center;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-top: 8px;
    margin-bottom: 16px;
}

/* Separador lateral */
.sidebar-section {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #4f9cf9;
    margin-top: 20px;
    margin-bottom: 6px;
}

/* Panel de detecciones */
.det-pill {
    display: inline-block;
    background: #1e2535;
    color: #c8d0e0;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 12px;
    margin: 2px;
}
.det-missing {
    color: #e06c75;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load pipeline module (cached for lifetime of server)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando pipeline…")
def load_pipeline():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline no encontrado: {PIPELINE_PATH}")
    spec = importlib.util.spec_from_file_location("cpak_pipeline", str(PIPELINE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar el módulo del pipeline")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner="Cargando modelo de detección…")
def load_det_model(path: str):
    return pipeline.YOLO(path)


@st.cache_resource(show_spinner="Cargando modelo de pose…")
def load_pose_model(path: str):
    return pipeline.YOLO(path)


pipeline = load_pipeline()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cv2_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


JOINT_POINT_INDEXES = {
    "Cadera": [0],
    "Rodilla": [1, 3, 4, 5, 6, 7],
    "Tobillo": [2],
}

POINT_LABELS = {
    0: "P0 Cabeza Femoral",
    1: "P1 Centro Rodilla",
    2: "P2 Centro Tobillo",
    3: "P3 Condilo Medial",
    4: "P4 Condilo Lateral",
    5: "P5 Plateau Medial",
    6: "P6 Plateau Lateral",
    7: "P7 Notch Femoral",
}


def draw_points_on_pil_image(pil_image: Image.Image, local_points: dict, joint_name: str) -> Image.Image:
    """
    Dibuja puntos sobre una imagen PIL.
    
    Args:
        pil_image: imagen PIL
        local_points: dict {idx: (x, y)} en coordenadas locales del crop
        joint_name: nombre de la articulación (para adaptar tamaño de marker)
    
    Returns:
        imagen PIL con puntos dibujados
    """
    img_np = np.array(pil_image)
    h, w = img_np.shape[:2]
    
    # Color de punto por defecto
    color_rgb = (0, 255, 0)  # Verde
    
    # Tamaño de marker adaptado por articulación
    radius = 3 if joint_name == "Rodilla" else 5
    thickness = 2 if joint_name == "Rodilla" else 3
    
    # Convertir a BGR para cv2.circle
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    for idx, (px, py) in local_points.items():
        px, py = int(px), int(py)
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(img_bgr, (px, py), radius, color_rgb, -1)  # Punto lleno
            cv2.circle(img_bgr, (px, py), radius, (0, 0, 255), thickness)  # Borde rojo
    
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def fmt(v, d=2):
    return "N/A" if v is None else f"{v:.{d}f}"


def cpak_color(cpak: str) -> str:
    if "Varo" in cpak:
        return "#e06c75"
    if "Valgo" in cpak:
        return "#61afef"
    return "#98c379"


def draw_outlined_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple,
    font_scale: float,
    line_thickness: int,
) -> None:
    x, y = int(origin[0]), int(origin[1])
    outline = max(2, line_thickness + 2)
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        outline,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        max(1, line_thickness),
        cv2.LINE_AA,
    )


def side_anchor(payload: dict, side: str, img_w: int, img_h: int) -> tuple:
    for det in payload.get("detections", {}).values():
        if det.get("joint") == "Rodilla" and det.get("side") == side:
            x1, y1, x2, y2 = [int(v) for v in det.get("xyxy", [0, 0, 0, 0])]
            ax = max(10, min(img_w - 10, x1 + 8))
            ay = max(40, min(img_h - 10, y1 - 10))
            return ax, ay

    default_x = int(img_w * (0.58 if side == "Der" else 0.08))
    default_y = int(img_h * 0.22)
    return max(10, min(img_w - 10, default_x)), max(40, min(img_h - 10, default_y))


def add_metrics_summary_overlay(
    canvas_bgr: np.ndarray,
    payload: dict,
    line_thickness: int,
    text_scale_multiplier: float = 1.0,
    der_offset_xy: tuple = (0, 0),
    izq_offset_xy: tuple = (0, 0),
) -> np.ndarray:
    canvas = canvas_bgr.copy()
    h, w = canvas.shape[:2]
    base_scale = max(0.55, min(1.0, w / 1800.0 + 0.35))
    metric_scale = max(0.35, min(4.0, base_scale * float(text_scale_multiplier)))
    line_gap = int(max(18, 24 * metric_scale))

    for side in ["Der", "Izq"]:
        metrics = payload.get("sides", {}).get(side, {}).get("metrics", {})
        if metrics.get("status") != "ok":
            continue

        lines = [
            f"aHKA: {fmt(metrics.get('aHKA'))}",
            f"HKA: {fmt(metrics.get('HKA'))}",
            f"LDFA: {fmt(metrics.get('LDFA'))}",
            f"MPTA: {fmt(metrics.get('MPTA'))}",
        ]
        x0, y0 = side_anchor(payload, side, w, h)
        off_x, off_y = der_offset_xy if side == "Der" else izq_offset_xy
        x0 = max(10, min(w - 10, int(x0 + off_x)))
        y0 = max(20, min(h - 10, int(y0 + off_y)))
        for i, txt in enumerate(lines):
            draw_outlined_text(
                canvas,
                txt,
                (x0, y0 + i * line_gap),
                metric_scale,
                line_thickness,
            )

    return canvas


def render(
    image_path: Path,
    payload: dict,
    boxes: bool,
    axes: bool,
    mikulicz: bool = False,
    line_thickness: int = None,
    show_metric_summary: bool = False,
    summary_text_scale: float = 1.0,
    der_offset_xy: tuple = (0, 0),
    izq_offset_xy: tuple = (0, 0),
) -> Image.Image:
    canvas = pipeline.render_overlay_from_payload(
        image_path=image_path,
        payload=payload,
        draw_boxes=boxes,
        draw_axes=axes,
        draw_mikulicz=mikulicz,
        line_thickness_override=line_thickness,
    )

    if show_metric_summary:
        canvas = add_metrics_summary_overlay(
            canvas_bgr=canvas,
            payload=payload,
            line_thickness=max(1, int(line_thickness or 2)),
            text_scale_multiplier=summary_text_scale,
            der_offset_xy=der_offset_xy,
            izq_offset_xy=izq_offset_xy,
        )

    return cv2_to_pil(canvas)


def get_joint_crop(image_path: Path, payload: dict, side: str, joint: str, margin_px: int = 30):
    """
    Recorta la zona de un joint específico con margen adicional.
    Retorna: (pil_image, dict_de_puntos_locales, (x1, y1), (w_crop, h_crop))
    donde dict_de_puntos_locales es {idx: (x_local, y_local), ...}
    """
    for det in payload.get("detections", {}).values():
        if det["joint"] == joint and det["side"] == side:
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
            x1 = max(0, x1 - margin_px)
            y1 = max(0, y1 - margin_px)
            x2 = min(w, x2 + margin_px)
            y2 = min(h, y2 + margin_px)
            crop = img[y1:y2, x1:x2]
            
            # Extraer puntos globales y proyectar a coordenadas locales del crop
            points_global = payload.get("sides", {}).get(side, {}).get("points", {})
            local_points = {}
            for idx_str, pt_global in points_global.items():
                try:
                    idx = int(idx_str)
                    px, py = pt_global
                    # Proyectar a coordenadas locales del crop
                    px_local = px - x1
                    py_local = py - y1
                    # Verificar que está dentro del crop
                    if 0 <= px_local < crop.shape[1] and 0 <= py_local < crop.shape[0]:
                        local_points[idx] = (px_local, py_local)
                except (ValueError, TypeError):
                    pass

            return cv2_to_pil(crop), local_points, (x1, y1), (crop.shape[1], crop.shape[0])
    return None, {}, None, None


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "payload" not in st.session_state:
    st.session_state.payload = None
    st.session_state.image_path = None
    st.session_state.ran_once = False
if "zoom_edit_mode" not in st.session_state:
    st.session_state.zoom_edit_mode = False
if "show_metric_summary" not in st.session_state:
    st.session_state.show_metric_summary = False
if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.warning(
        "**Demo de investigación.** No subas datos identificables de pacientes "
        "(nombres, RUT, fechas de nacimiento). No es un dispositivo médico "
        "certificado: los resultados deben ser revisados por un cirujano."
    )

    st.markdown('<div class="sidebar-section">Imagen</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Seleccionar Rx",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        label_visibility="collapsed",
    )
    st.caption("💡 También puedes pegar una imagen con **Ctrl+V** (p. ej. un pantallazo).")

    # Escucha el pegado desde el portapapeles e inyecta la imagen en el uploader.
    components.html(PASTE_HANDLER_HTML, height=0)

    st.markdown('<div class="sidebar-section">Modelos</div>', unsafe_allow_html=True)
    det_path = st.text_input("Detección (.pt)", value=str(pipeline.DEFAULT_DET_MODEL))
    pose_path = st.text_input("Pose (.pt)", value=str(pipeline.DEFAULT_POSE_MODEL))

    st.markdown('<div class="sidebar-section">Parámetros</div>', unsafe_allow_html=True)
    det_conf = st.slider("Detection conf", 0.0, 1.0, 0.20, 0.05)
    pose_conf = st.slider("Pose conf", 0.0, 1.0, 0.20, 0.05)
    padding = st.slider("Padding BB (%)", 0.0, 0.50, 0.15, 0.05)

    st.markdown('<div class="sidebar-section">Capas de visualización</div>', unsafe_allow_html=True)
    show_boxes = st.checkbox("Bounding Boxes", value=True)
    show_axes = st.checkbox("Ejes anatómicos", value=True)
    show_mikulicz = st.checkbox("Línea de Mikulicz", value=False)
    line_thickness = st.slider("Grosor líneas", 1, 10, 2, 1)
    summary_text_scale = st.slider("Tamaño métricas resumen", 0.4, 3.5, 1.0, 0.05)
    der_offset_x = st.slider("Mover resumen Der X", -400, 400, 400, 5)
    der_offset_y = st.slider("Mover resumen Der Y", -400, 400, 5, 5)
    izq_offset_x = st.slider("Mover resumen Izq X", -400, 400, -400, 5)
    izq_offset_y = st.slider("Mover resumen Izq Y", -400, 400, -400, 5)

    st.markdown("---")
    run_btn = st.button("▶  Ejecutar inferencia", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="cpak-title">🦴 CPAK · Inferencia Clínica</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cpak-subtitle">Detección de zonas · Keypoints anatómicos · LDFA / MPTA / aHKA / JLO</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
if run_btn:
    if uploaded is None:
        st.error("Carga una imagen Rx primero.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        try:
            with st.spinner("Ejecutando pipeline…"):
                det_model = load_det_model(det_path)
                pose_model = load_pose_model(pose_path)
                out_dir = Path(tempfile.gettempdir()) / "cpak_streamlit"
                _, _, payload = pipeline.process_image(
                    det_model=det_model,
                    pose_model=pose_model,
                    image_path=tmp_path,
                    output_dir=out_dir,
                    det_conf=det_conf,
                    pose_conf=pose_conf,
                    padding=padding,
                    draw_boxes=show_boxes,
                    draw_axes=show_axes,
                )
            st.session_state.payload = payload
            st.session_state.image_path = tmp_path
            st.session_state.ran_once = True
            st.success("Inferencia completada ✓")
        except Exception as exc:
            st.error(f"Error: {exc}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if st.session_state.ran_once and st.session_state.payload is not None:
    payload = st.session_state.payload
    img_path = st.session_state.image_path

    # Re-render on every interaction (instant, no re-inference)
    overlay = render(
        img_path,
        payload,
        show_boxes,
        show_axes,
        show_mikulicz,
        line_thickness=line_thickness,
        show_metric_summary=st.session_state.show_metric_summary,
        summary_text_scale=summary_text_scale,
        der_offset_xy=(der_offset_x, der_offset_y),
        izq_offset_xy=(izq_offset_x, izq_offset_y),
    )

    # ── Layout: métricas | imagen | controles ──
    col_left, col_center, col_right = st.columns([1, 2.2, 1], gap="large")

    # ── Métricas ──────────────────────────────────────────────────────────
    with col_left:
        st.markdown("### 📊 Métricas")

        for side in ["Der", "Izq"]:
            side_color = "#98c379" if side == "Der" else "#61afef"
            st.markdown(
                f'<span style="font-size:16px;font-weight:700;color:{side_color};">{side}</span>',
                unsafe_allow_html=True,
            )
            s = payload.get("sides", {}).get(side, {})
            m = s.get("metrics", {})

            # Compatibilidad con payloads antiguos: recalcula HKA si no viene en métricas.
            if m.get("status") == "ok" and m.get("HKA") is None:
                pts_raw = s.get("points", {})
                pts = {
                    int(k): (float(v[0]), float(v[1]))
                    for k, v in pts_raw.items()
                    if isinstance(v, (list, tuple)) and len(v) >= 2
                }
                if all(idx in pts for idx in [0, 1, 2, 7]):
                    hka_val = pipeline.line_angle_deg(pts[0], pts[7], pts[1], pts[2])
                    if hka_val is not None:
                        m["HKA"] = round(float(hka_val), 3)

            if m.get("status") == "ok":
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("LDFA °", fmt(m.get("LDFA")))
                    st.metric("aHKA °", fmt(m.get("aHKA")))
                with c2:
                    st.metric("MPTA °", fmt(m.get("MPTA")))
                    st.metric("HKA °", fmt(m.get("HKA")))
                    st.metric("JLO °", fmt(m.get("JLO")))
                
                mad_px = m.get("MAD_px")
                if mad_px is not None:
                    st.metric("MAD (px)", f"{mad_px:.1f}", help="Desviación de eje mecánico: distancia perpendicular desde centro de rodilla a línea Mikulicz")

                cpak = m.get("CPAK", "")
                cpak_type = m.get("CPAK_type", "?")
                color = cpak_color(cpak)
                st.markdown(
                    f'<div class="cpak-badge" style="background:{color}22;'
                    f'border:1.5px solid {color};color:{color};">'
                    f'Tipo {cpak_type} &nbsp;·&nbsp; {cpak}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning(f"⚠ {m.get('status', 'datos incompletos')}")

    # ── Imagen central ─────────────────────────────────────────────────────
    with col_center:
        st.markdown("### 🖼️ Overlay")
        st.image(overlay, use_container_width=True)

        # Zoom por articulación
        with st.expander("🔍 Zoom por articulación", expanded=st.session_state.get("zoom_edit_mode", False)):
            z_side = st.radio("Lado", ["Der", "Izq"], horizontal=True, key="zoom_side")
            z_joint = st.radio("Articulación", ["Cadera", "Rodilla", "Tobillo"], horizontal=True, key="zoom_joint")
            z_margin = st.slider("Margen (px)", 10, 200, 60, 10, key="zoom_margin")

            crop, local_points, crop_origin, crop_size = get_joint_crop(img_path, payload, z_side, z_joint, z_margin)
            if crop:
                # Edición por articulación en el mismo panel de zoom
                edit_mode = st.toggle(
                    "✏️ Editar puntos de esta articulación",
                    key="zoom_edit_mode",
                )

                # Dibujar puntos sobre el crop
                crop_with_points = draw_points_on_pil_image(crop, local_points, z_joint)

                # Usar el ancho disponible del panel (se escalará con use_container_width)
                crop_w, crop_h = crop.size
                display_w = min(1600, crop_w)
                display_scale = display_w / max(1, crop_w)
                display_h = max(1, int(crop_h * display_scale))
                crop_display = crop.resize((display_w, display_h), Image.Resampling.BILINEAR)
                crop_points_display = crop_with_points.resize((display_w, display_h), Image.Resampling.BILINEAR)

                if edit_mode and crop_origin and crop_size:
                    target_indices = JOINT_POINT_INDEXES.get(z_joint, [])
                    x0, y0 = crop_origin

                    if not CANVAS_AVAILABLE:
                        st.warning(
                            "Para edición drag en la app, instala: pip install streamlit-drawable-canvas"
                        )
                    else:
                        st.caption("La imagen es la misma del zoom. Arrastra puntos y guarda.")

                        state_key = f"canvas_init_{z_side}_{z_joint}_{z_margin}_{Path(img_path).name}"
                        if state_key not in st.session_state:
                            objects = []
                            for idx in target_indices:
                                lx, ly = local_points.get(idx, (crop_w / 2.0, crop_h / 2.0))
                                zx = float(lx * display_scale)
                                zy = float(ly * display_scale)
                                r = 6.0
                                objects.append(
                                    {
                                        "type": "circle",
                                        "left": zx - r,
                                        "top": zy - r,
                                        "radius": r,
                                        "fill": "rgba(0,255,0,0.45)",
                                        "stroke": "#ff3333",
                                        "strokeWidth": 2,
                                        "name": str(idx),
                                    }
                                )
                            st.session_state[state_key] = {"version": "4.4.0", "objects": objects}

                        canvas_result = st_canvas(
                            fill_color="rgba(0,255,0,0.3)",
                            stroke_width=2,
                            stroke_color="#ff3333",
                            background_image=crop_display,
                            update_streamlit=True,
                            height=display_h,
                            width=display_w,
                            drawing_mode="transform",
                            initial_drawing=st.session_state[state_key],
                            key=f"canvas_{z_side}_{z_joint}_{z_margin}_{Path(img_path).name}_v{st.session_state.canvas_version}",
                        )

                        # NO persistimos el estado automáticamente: el canvas mantiene su estado interno.
                        # Solo se guarda cuando el usuario presiona "Guardar puntos" abajo.
                        # Esto evita el loop de re-render infinito.

                        if st.button("💾 Guardar puntos desde drag", key=f"save_canvas_{z_side}_{z_joint}"):
                            # El canvas_result puede estar vacío en este render (el botón no
                            # es una interacción del canvas). Leemos del session_state del
                            # widget, que retiene el último estado enviado por el canvas (drag).
                            canvas_widget_key = f"canvas_{z_side}_{z_joint}_{z_margin}_{Path(img_path).name}_v{st.session_state.canvas_version}"
                            raw_state = st.session_state.get(canvas_widget_key)

                            jdata = {}
                            if isinstance(raw_state, dict):
                                # El widget devuelve un dict con 'raw' que contiene los objetos de Fabric.js
                                raw = raw_state.get('raw')
                                if isinstance(raw, dict):
                                    jdata = raw
                                elif isinstance(raw_state, dict) and 'objects' in raw_state:
                                    jdata = raw_state
                            elif raw_state is not None and hasattr(raw_state, 'json_data') and raw_state.json_data is not None:
                                jdata = raw_state.json_data

                            objects = jdata.get("objects", []) if isinstance(jdata, dict) else []
                            if not objects:
                                objects = st.session_state.get(state_key, {}).get("objects", [])
                            if not objects:
                                st.warning("No hay puntos detectados en el canvas.")
                            else:
                                # Crear copia profunda del payload para garantizar que Streamlit detecte el cambio
                                import copy
                                new_payload = copy.deepcopy(st.session_state.payload)
                                side_points = new_payload.setdefault("sides", {}).setdefault(z_side, {}).setdefault("points", {})

                                for obj in objects:
                                    try:
                                        idx = int(obj.get("name"))
                                    except (TypeError, ValueError):
                                        continue
                                    if idx not in target_indices:
                                        continue
                                    # Centro del círculo en canvas coords → local crop → global imagen
                                    cx = float(obj.get("left", 0.0)) + float(obj.get("radius", 6.0))
                                    cy = float(obj.get("top", 0.0)) + float(obj.get("radius", 6.0))
                                    lx = max(0.0, min(crop_w - 1.0, cx / display_scale))
                                    ly = max(0.0, min(crop_h - 1.0, cy / display_scale))
                                    gx = float(x0 + lx)
                                    gy = float(y0 + ly)
                                    side_points[str(idx)] = [round(gx, 2), round(gy, 2)]

                                # Recalcular métricas
                                all_pts = new_payload.get("sides", {}).get(z_side, {}).get("points", {})
                                pts_for_calc = {int(k): v for k, v in all_pts.items()}
                                new_metrics = pipeline.calculate_metrics(pts_for_calc)
                                new_payload["sides"][z_side]["metrics"] = new_metrics

                                # Guardar
                                st.session_state.payload = new_payload

                                # Sincronizar initial_drawing
                                new_objects = []
                                r = 6.0
                                for idx_str, (gx2, gy2) in side_points.items():
                                    idx = int(idx_str)
                                    lx2 = gx2 - x0
                                    ly2 = gy2 - y0
                                    zx2 = float(lx2 * display_scale)
                                    zy2 = float(ly2 * display_scale)
                                    new_objects.append({
                                        "type": "circle",
                                        "left": zx2 - r,
                                        "top": zy2 - r,
                                        "radius": r,
                                        "fill": "rgba(0,255,0,0.45)",
                                        "stroke": "#ff3333",
                                        "strokeWidth": 2,
                                        "name": str(idx),
                                    })
                                st.session_state[state_key] = {"version": "4.4.0", "objects": new_objects}
                                st.session_state.canvas_version += 1
                                st.success("✅ Puntos guardados y métricas recalculadas.")
                                st.rerun()
                else:
                    st.image(
                        crop_points_display,
                        caption=f"{z_side} · {z_joint} ({len(local_points)} pts)",
                        use_container_width=True,
                    )
            else:
                st.info(f"No se detectó {z_joint} ({z_side})")

    # ── Panel derecho ───────────────────────────────────────────────────────
    with col_right:
        st.markdown("### 🎛️ Info")

        summary_label = "📝 Mostrar resumen en imagen"
        if st.session_state.show_metric_summary:
            summary_label = "📝 Ocultar resumen en imagen"
        if st.button(summary_label, use_container_width=True):
            st.session_state.show_metric_summary = not st.session_state.show_metric_summary
            st.rerun()

        # Detecciones encontradas
        st.markdown("**Detecciones**")
        JOINTS = ["Cadera", "Rodilla", "Tobillo"]
        for side in ["Der", "Izq"]:
            found = payload.get("sides", {}).get(side, {}).get("detections_found", [])
            pills = ""
            for j in JOINTS:
                cls = "det-pill" if j in found else "det-pill det-missing"
                icon = "✓" if j in found else "✗"
                pills += f'<span class="{cls}">{icon} {j}</span>'
            st.markdown(
                f'<div style="margin-bottom:6px;"><b style="font-size:12px;">{side}</b><br>{pills}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Descargas
        st.markdown("**Descargas**")

        json_bytes = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "📥 JSON métricas",
            data=json_bytes,
            file_name="cpak_metrics.json",
            mime="application/json",
            use_container_width=True,
        )

        img_bytes = BytesIO()
        overlay.save(img_bytes, format="JPEG", quality=92)
        st.download_button(
            "🖼️ Overlay JPG",
            data=img_bytes.getvalue(),
            file_name="cpak_overlay.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("**JSON completo**")
        with st.expander("Ver payload"):
            st.json(payload, expanded=False)

else:
    # Estado inicial
    st.markdown(
        """
        <div style="text-align:center;padding:80px 0;color:#8b9ab1;">
            <div style="font-size:60px;">🦴</div>
            <div style="font-size:18px;margin-top:16px;">Carga una imagen Rx y presiona <b>Ejecutar inferencia</b></div>
            <div style="font-size:13px;margin-top:8px;">Usa el panel lateral para configurar los modelos y parámetros</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
