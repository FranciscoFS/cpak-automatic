"""Interfaz del simulador de cortes, reutilizable dentro de otra app Streamlit.

Expone dos cosas:
    inject_theme()                      -> CSS del estilo "claro" institucional
    render_simulator(image, payload)    -> dibuja la vista del simulador

La geometria vive en `cut_simulator.py`; aqui solo hay interfaz.
"""

import base64
import importlib.util
import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:  # noqa: BLE001
    CANVAS_AVAILABLE = False

BONE_COLORS = {"Tibia": (226, 125, 48), "Femur": (58, 131, 232)}  # BGR

_THEME_CSS = """
<style>
html, body, [class*="css"], .stApp {
    font-family: "Inter", "Segoe UI Variable", "Segoe UI", system-ui, -apple-system, sans-serif;
}
.stApp { background: #f6f7f9; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stDecoration"] { display: none; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stMainBlockContainer"] { padding: 1.1rem .8rem 3rem .8rem; max-width: 1600px; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8eaee; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }

.cpak-hero { margin: 0 0 1.1rem 0; }
.cpak-hero-title { font-size: 1.6rem; font-weight: 700; letter-spacing: -.02em; color: #0f172a; }
.cpak-hero-sub { font-size: .87rem; color: #6b7280; margin-top: .18rem; }
.cpak-brand { font-size: .78rem; font-weight: 700; letter-spacing: .16em;
    text-transform: uppercase; color: #307de2; margin-bottom: .6rem; }
.cpak-sect { font-size: .69rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: #9aa3b2; margin: 1.05rem 0 .25rem; }

[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e8eaee; border-radius: 14px;
    padding: .8rem .95rem; box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}
[data-testid="stMetricLabel"] p {
    font-size: .68rem !important; text-transform: uppercase;
    letter-spacing: .07em; color: #8a93a3 !important; font-weight: 600;
}
[data-testid="stMetricValue"] {
    font-size: 1.45rem !important; font-weight: 650 !important; color: #0f172a;
}
[data-testid="stImage"] img { border-radius: 14px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
.stButton > button, .stDownloadButton > button {
    border-radius: 999px; font-weight: 600; border: 1px solid #e2e6ec;
}
</style>
"""


def inject_theme() -> None:
    """Inyecta el estilo visual claro usado en el simulador."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def load_geometry():
    """Carga `cut_simulator.py` (geometria pura, sin Streamlit)."""
    path = Path(__file__).with_name("cut_simulator.py")
    if not path.exists():
        raise FileNotFoundError(f"No se encontro {path}")
    spec = importlib.util.spec_from_file_location("cpak_cut_simulator", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar cut_simulator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bgr_to_pil(canvas: np.ndarray, max_width: int = 1400):
    from PIL import Image

    height, width = canvas.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        canvas = cv2.resize(canvas, (max_width, int(round(height * scale))),
                            interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def _inline_image_html(canvas: np.ndarray, display_w: int, quality: int = 90) -> str:
    """Imagen como data-URI embebido.

    Evita el endpoint /media/ de Streamlit, que en Streamlit Cloud responde 500
    con overlays grandes y deja la imagen rota.
    """
    pil = _bgr_to_pil(canvas, int(display_w))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return (
        f'<img src="data:image/jpeg;base64,{b64}" '
        f'style="width:100%;max-width:{int(display_w)}px;height:auto;'
        f'border-radius:10px;display:block;margin:0 auto;" />'
    )


def _crop(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]


@st.dialog("Calibración", width="large")
def _calibration_dialog(img, pts, scale_mm_per_px, source, active_bone, sim_lib):
    st.caption(
        "El ancho entre los dos puntos marcados es lo que se declara en milímetros. "
        "La escala resultante se aplica a los dos cortes."
    )
    for bone in ("Tibia", "Femur"):
        try:
            seg = sim_lib.reference_segment(pts, bone)
        except ValueError:
            st.warning(f"{bone}: faltan puntos para medir su ancho.")
            continue
        p_a = np.array(seg["p_med"], dtype=float)
        p_b = np.array(seg["p_lat"], dtype=float)
        xs, ys = [p_a[0], p_b[0]], [p_a[1], p_b[1]]
        margin = 0.40 * max(abs(p_a[0] - p_b[0]), abs(p_a[1] - p_b[1]), 60.0)
        x0 = int(max(0, min(xs) - margin))
        x1 = int(min(img.shape[1], max(xs) + margin))
        y0 = int(max(0, min(ys) - margin))
        y1 = int(min(img.shape[0], max(ys) + margin))
        crop = img[y0:y1, x0:x1].copy()
        if not crop.size:
            continue
        target_w = 620.0
        f_dlg = target_w / float(crop.shape[1])
        crop = cv2.resize(
            crop,
            (int(round(crop.shape[1] * f_dlg)), int(round(crop.shape[0] * f_dlg))),
            interpolation=cv2.INTER_AREA if f_dlg < 1.0 else cv2.INTER_LINEAR,
        )
        a = (int(round((p_a[0] - x0) * f_dlg)), int(round((p_a[1] - y0) * f_dlg)))
        b = (int(round((p_b[0] - x0) * f_dlg)), int(round((p_b[1] - y0) * f_dlg)))
        color = BONE_COLORS[bone]
        cv2.line(crop, a, b, color, max(2, int(round(crop.shape[1] / 200.0))), cv2.LINE_AA)
        rad = max(4, int(round(crop.shape[1] / 70.0)))
        for p in (a, b):
            cv2.circle(crop, p, rad, color, 2, cv2.LINE_AA)
            cv2.circle(crop, p, max(2, rad // 3), color, -1, cv2.LINE_AA)
        label = f"{seg['width_px']:.1f} px"
        if scale_mm_per_px:
            label += f"  =  {seg['width_px'] * scale_mm_per_px:.1f} mm"
        f_txt = crop.shape[1] / 620.0
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * f_txt, 2)
        tx, ty = int(mid[0] - tw / 2), int(min(a[1], b[1]) - 18 * f_txt)
        box = crop.copy()
        cv2.rectangle(box, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8), (25, 28, 35), -1)
        crop = cv2.addWeighted(box, 0.72, crop, 0.28, 0)
        cv2.putText(crop, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * f_txt,
                    (255, 255, 255), max(1, int(round(2 * f_txt))), cv2.LINE_AA)
        mark = " · referencia" if bone == active_bone else ""
        st.markdown(f"**{bone}**{mark}")
        st.html(_inline_image_html(crop, int(target_w)))

    if scale_mm_per_px:
        st.success(f"Escala en uso: {scale_mm_per_px:.4f} mm/px"
                   + (f" · {source}" if source else ""))
    else:
        st.info("Sin escala: las cifras están en píxeles.")


def render_simulator(image: np.ndarray, payload: dict, mm_per_px: Optional[float] = None,
                     key: str = "sim") -> None:
    """Vista del simulador de cortes sobre una imagen y unos puntos ya cargados."""
    sim_lib = load_geometry()
    sides_found = (payload or {}).get("sides", {})
    if not sides_found:
        st.info("Corre la inferencia para obtener los puntos.")
        return

    # --------------------- layout: imagen | controles ---------------------
    col_img, col_ctrl = st.columns([2.2, 1], gap="large")

    with col_ctrl:
        side_opts = list(sides_found.keys())
        side = st.pills("Lado", side_opts, default=side_opts[0],
                        label_visibility="collapsed", key=f"{key}_side") or side_opts[0]
        c_inv, c_ax = st.columns(2)
        invert_ml = c_inv.toggle("Invertir M/L", value=False, key=f"{key}_inv")
        show_axes = c_ax.toggle("Ejes", value=False, key=f"{key}_axes",
                                help="Eje mecanico de cada hueso en gris punteado.")
        open_calib = st.button("Calibracion", icon="📏", width="stretch",
                               key=f"{key}_calib_btn",
                               help="Ver el recorte con los puntos que definen la escala.")
        pivot_mode = "Mecanico"

        with st.expander("Escala y visualizacion", expanded=False):
            if mm_per_px:
                st.caption(f"DICOM · {mm_per_px:.4f} mm/px")
                use_mm, ref_bone, ref_width_mm = True, None, None
            else:
                use_mm = st.toggle("Escala aproximada", value=True, key=f"{key}_usemm")
                ref_bone = (st.radio("Hueso de referencia", ("Tibia", "Fémur"),
                                     horizontal=True, key=f"{key}_refbone") if use_mm else None)
                ref_width_mm = (st.number_input("Ancho de referencia (mm)", 20.0, 140.0,
                                                80.0, 1.0, key=f"{key}_refw") if use_mm else None)
            style_label = st.selectbox(
                "Estilo",
                ("Línea (radiografía)", "Limpio (solo líneas)",
                 "Blueprint (gris suave)", "Diagrama (fondo blanco)"),
                key=f"{key}_style",
            )
            style = {"Línea (radiografía)": "line", "Limpio (solo líneas)": "clean",
                     "Blueprint (gris suave)": "blueprint",
                     "Diagrama (fondo blanco)": "diagram"}[style_label]
            thickness = st.slider("Grosor de líneas", 1, 6, 2, 1, key=f"{key}_thick")
            display_w = st.slider("Tamaño en pantalla (px)", 480, 1400, 980, 20,
                                  key=f"{key}_disp")
            zoom = st.slider("Encuadre", 1.0, 3.0, 2.0, 0.1, key=f"{key}_zoom")
            layer_alpha = st.slider("Opacidad de capas", 0.05, 0.70, 0.25, 0.05,
                                    key=f"{key}_alpha")
            show_prox = st.toggle("Hueso proximal", value=False, key=f"{key}_prox")
            show_dist = st.toggle("Hueso distal", value=False, key=f"{key}_dist")
            show_res = st.toggle("Hueso resecado", value=False, key=f"{key}_res")

        st.markdown(
            '<div style="font-size:.72rem;color:#8a93a3;margin:.2rem 0 .6rem;">'
            '0° = perpendicular al eje mecánico · '
            '<b>−</b> varo · <b>+</b> valgo</div>',
            unsafe_allow_html=True,
        )

        unit = "mm" if use_mm else "px"
        cut_settings = {}
        for bone in ("Tibia", "Femur"):
            with st.container(border=True):
                visible = st.toggle(bone, value=True, key=f"{key}_on_{bone}")
                c_ang, c_dep = st.columns(2, gap="small")
                angle = c_ang.slider("Ángulo (°)", -15.0, 15.0, 5.0, 0.5,
                                     key=f"{key}_ang_{bone}")
                c_ang.markdown(
                    '<div style="display:flex;justify-content:space-between;'
                    'font-size:.68rem;color:#8a93a3;margin-top:-.65rem;">'
                    '<span>← Varo</span><span>Valgo →</span></div>',
                    unsafe_allow_html=True,
                )
                if unit == "mm":
                    dep = c_dep.slider("Prof. (mm)", 0.0, 30.0, 8.0, 0.5,
                                       key=f"{key}_dep_{bone}")
                else:
                    dep = c_dep.slider("Prof. (px)", 0.0, 400.0, 100.0, 5.0,
                                       key=f"{key}_dep_{bone}")
            cut_settings[bone] = {"visible": visible, "angle": angle, "depth": dep}

    points = {k: list(v) for k, v in (sides_found.get(side, {}) or {}).get("points", {}).items()}
    if not points:
        st.error(f"No hay puntos para el lado {side}.")
        return

    # Ajustes manuales
    edit_id = f"{key}_{side}"
    if "sim_point_overrides" not in st.session_state:
        st.session_state.sim_point_overrides = {}
    for idx, coords in (st.session_state.sim_point_overrides.get(edit_id) or {}).items():
        if str(idx) in points:
            points[str(idx)] = [float(coords[0]), float(coords[1])]

    # Escala compartida
    scale_mm_per_px = mm_per_px
    calibration_source = "PixelSpacing (DICOM)" if mm_per_px else None
    if scale_mm_per_px is None and use_mm and ref_bone and ref_width_mm:
        try:
            ref_seg = sim_lib.reference_segment(points, ref_bone, invert_ml=invert_ml)
            if ref_seg["width_px"] > 0:
                scale_mm_per_px = float(ref_width_mm) / ref_seg["width_px"]
                calibration_source = (f"ancho {ref_bone.lower()} "
                                      f"{float(ref_width_mm):.0f} mm (compartida)")
        except ValueError as exc:
            st.warning(f"No se pudo calibrar con {ref_bone}: {exc}")

    if open_calib:
        st.session_state[f"{key}_calib_requested"] = True
    if st.session_state.pop(f"{key}_calib_requested", False):
        _calibration_dialog(image, points, scale_mm_per_px, calibration_source,
                            ref_bone, sim_lib)

    sims = []
    for bone, cfg in cut_settings.items():
        if not cfg["visible"]:
            continue
        req = sim_lib.CUT_TYPES[bone]
        missing = [i for i in (req["med_idx"], req["lat_idx"]) if str(i) not in points]
        if missing:
            st.warning(f"{bone}: faltan los puntos {missing}. Se omite este corte.")
            continue
        kwargs = ({"offset_mm": float(cfg["depth"])} if unit == "mm"
                  else {"offset_px": float(cfg["depth"])})
        try:
            sims.append((bone, sim_lib.simulate_cut(
                points=points, cut_type=bone, angle_deg=cfg["angle"],
                pivot_mode=pivot_mode, invert_ml=invert_ml, side=side,
                mm_per_px=scale_mm_per_px, calibration_source=calibration_source,
                **kwargs)))
        except ValueError as exc:
            st.warning(f"{bone}: no se pudo simular ({exc}).")

    if not sims:
        with col_img:
            st.error("Activa al menos un hueso con puntos validos.")
        return

    sim = sims[0][1]
    painted, bone_stats = sim_lib.paint_bone_regions(
        image, sim, show_proximal=show_prox, show_distal=show_dist,
        show_resected=show_res, alpha=layer_alpha,
    )

    crop_ref = dict(sims[0][1])
    if len(sims) > 1:
        for k in ("p_med", "p_lat", "proj_med", "proj_lat"):
            avg = np.mean([np.array(s[k], dtype=float) for _, s in sims], axis=0)
            crop_ref[k] = (float(avg[0]), float(avg[1]))
        crop_ref["width_px"] = float(np.linalg.norm(
            np.array(crop_ref["p_lat"]) - np.array(crop_ref["p_med"])))

    zoom_x = min(zoom, 1.4) if style == "diagram" else zoom
    zoom_y = zoom_x * 0.62
    bx0, by0, bx1, by1 = sim_lib.crop_bounds(crop_ref, image.shape, zoom_x, zoom_y)
    crop = _crop(painted, bx0, by0, bx1, by1)

    with col_img:
        if crop.size == 0:
            st.error("El recorte quedo vacio; ajusta el encuadre.")
            return
        factor = min(float(display_w) / crop.shape[1], 6.0)
        if abs(factor - 1.0) > 1e-3:
            crop = cv2.resize(
                crop,
                (int(round(crop.shape[1] * factor)), int(round(crop.shape[0] * factor))),
                interpolation=cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC,
            )
            factor = crop.shape[1] / float(max(1, bx1 - bx0))

        disp_sims = [(b, sim_lib.scale_sim(s, bx0, by0, factor)) for b, s in sims]
        line_w = max(1, int(round(thickness * crop.shape[1] / 900.0)))
        overlay = crop
        if show_axes:
            overlay = sim_lib.draw_mechanical_axes(
                overlay, [s for _, s in disp_sims], thickness=max(1, line_w - 1))
        for bone, ds in disp_sims:
            overlay = sim_lib.draw_cut_overlay(
                overlay, ds, line_thickness=line_w,
                fill_wedge=(not show_res) and len(sims) == 1,
                style=style, show_values=False, color=BONE_COLORS[bone],
            )
        overlay = sim_lib.draw_hud(
            overlay, [(b.upper(), s, BONE_COLORS[b]) for b, s in disp_sims],
            badge=f"{side}  /  {' + '.join(b for b, _ in sims)}",
        )
        st.html(_inline_image_html(overlay, int(display_w)))

        if not scale_mm_per_px:
            st.caption("Sin calibracion milimetrica: las cifras estan en pixeles.")
        elif calibration_source and "compartida" in str(calibration_source):
            st.caption(f"Calibracion aproximada: {calibration_source} · "
                       f"{scale_mm_per_px:.4f} mm/px aplicada a los dos cortes.")

    # ------------------------------ detalle ------------------------------
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        with st.expander("Tabla de medidas", expanded=False):
            rows = []
            for bone, bone_sim in sims:
                for k, v in sim_lib.summarize(bone_sim, bone_stats if len(sims) == 1 else None):
                    rows.append({"Hueso": bone, "Concepto": k, "Valor": v})
            st.dataframe(rows, width="stretch", hide_index=True)
    with col_b:
        with st.expander("Ajustar puntos", expanded=False):
            groups = {"Rodilla (P1, P3–P7)": [1, 3, 4, 5, 6, 7],
                      "Cadera (P0)": [0], "Tobillo (P2)": [2]}
            group = st.radio("Zona", list(groups), horizontal=True, key=f"{key}_zone",
                             label_visibility="collapsed")
            idxs = [i for i in groups[group] if str(i) in points]
            if idxs and CANVAS_AVAILABLE:
                xs = [points[str(i)][0] for i in idxs]
                ys = [points[str(i)][1] for i in idxs]
                span = max(max(xs) - min(xs), max(ys) - min(ys), 60.0)
                m = 0.95 * span
                x0 = int(max(0, min(xs) - m))
                x1 = int(min(image.shape[1], max(xs) + m))
                y0 = int(max(0, min(ys) - m))
                y1 = int(min(image.shape[0], max(ys) + m))
                sub = _crop(image, x0, y0, x1, y1)
                if sub.size:
                    cw = 420
                    sc = cw / float(sub.shape[1])
                    bg = cv2.resize(sub, (cw, int(round(sub.shape[0] * sc))),
                                    interpolation=cv2.INTER_AREA)
                    rad = 8
                    objs = [{
                        "type": "circle",
                        "left": (points[str(i)][0] - x0) * sc - rad,
                        "top": (points[str(i)][1] - y0) * sc - rad,
                        "radius": rad, "fill": "rgba(48,125,226,0.30)",
                        "stroke": "#307DE2", "strokeWidth": 2, "name": str(i),
                    } for i in idxs]
                    canvas = None
                    try:
                        canvas = st_canvas(
                            fill_color="rgba(48,125,226,0.30)", stroke_width=2,
                            stroke_color="#307DE2",
                            background_image=_bgr_to_pil(bg, cw),
                            update_streamlit=True, height=int(bg.shape[0]), width=int(cw),
                            drawing_mode="transform",
                            initial_drawing={"version": "4.4.0", "objects": objs},
                            key=f"{key}_canvas_{edit_id}_{group}",
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.caption(
                            f"Arrastre no disponible con esta version de "
                            f"streamlit-drawable-canvas ({type(exc).__name__}). "
                            "Ajusta las coordenadas en la tabla."
                        )
                        plain = st.data_editor(
                            [{"Punto": f"P{i}",
                              "x (px)": round(float(points[str(i)][0]), 1),
                              "y (px)": round(float(points[str(i)][1]), 1)} for i in idxs],
                            hide_index=True, width="stretch",
                            key=f"{key}_nums_{edit_id}_{group}",
                        )
                        if st.button("Aplicar coordenadas", width="stretch",
                                     key=f"{key}_applynum_{group}"):
                            store = st.session_state.sim_point_overrides.setdefault(edit_id, {})
                            for row in plain:
                                idx = int(str(row["Punto"]).lstrip("P"))
                                store[idx] = (float(row["x (px)"]), float(row["y (px)"]))
                            st.rerun()
                    if canvas is not None:
                        st.caption("Arrastra los círculos · " + " · ".join(f"P{i}" for i in idxs))
                        c_ap, c_rs = st.columns(2)
                        if c_ap.button("Aplicar", width="stretch", key=f"{key}_apply_{group}"):
                            store = st.session_state.sim_point_overrides.setdefault(edit_id, {})
                            for idx, obj in zip(idxs, (canvas.json_data or {}).get("objects", [])):
                                r_obj = float(obj.get("radius", rad))
                                store[idx] = ((obj["left"] + r_obj) / sc + x0,
                                              (obj["top"] + r_obj) / sc + y0)
                            st.rerun()
                        if c_rs.button("Restablecer", width="stretch", key=f"{key}_reset_{group}"):
                            st.session_state.sim_point_overrides.pop(edit_id, None)
                            st.rerun()
