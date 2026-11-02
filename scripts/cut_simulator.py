"""Simulador geometrico de cortes oseos sobre telerradiografias CPAK.

Geometria 2D en pixeles de imagen (x a la derecha, y hacia abajo). El modulo no
depende de Streamlit ni de YOLO: recibe los ocho puntos P0-P7 y devuelve las
medidas del corte. `scripts/31_cut_simulator_app.py` lo usa como prototipo local;
la idea es integrarlo despues en `scripts/30_streamlit_app.py`.

Convenciones
------------
- Linea de referencia = linea articular: P5-P6 para tibia, P3-P4 para femur.
- El plano de corte pasa por un pivote sobre esa linea, gira `angle_deg`
  respecto de ella y se puede desplazar `offset_px` hacia distal.
  0 grados = corte paralelo a la articulacion; offset 0 = pegado a la linea
  articular.
- Las distancias son perpendiculares al plano de corte. Se informan en pixeles
  y, si hay calibracion, en milimetros. La calibracion puede venir del
  PixelSpacing del DICOM (preferente) o de un ancho articular aproximado
  declarado por el usuario. Nunca se inventa una escala.
"""

import math
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

CUT_TYPES = {
    "Tibia": {"med_idx": 5, "lat_idx": 6, "reference": "Platillo tibial (P5-P6)"},
    "Femur": {"med_idx": 3, "lat_idx": 4, "reference": "Condilos femorales (P3-P4)"},
}
PIVOT_MODES = ("Mecanico", "Medial", "Centro", "Lateral")

CUT_COLOR = (255, 0, 255)        # BGR magenta: plano de corte
REFERENCE_COLOR = (170, 170, 170)  # linea articular de referencia
MEASURE_COLOR = (60, 200, 60)    # distancias perpendiculares
WEDGE_COLOR = (255, 140, 60)     # cuna simulada
PROXIMAL_COLOR = (255, 130, 0)   # hueso que queda proximal (arriba)
DISTAL_COLOR = (0, 150, 255)     # hueso que queda distal (abajo)
RESECTED_COLOR = (0, 0, 255)     # hueso que se reseca



# ---------------------------------------------------------------------------
# Utilidades geometricas
# ---------------------------------------------------------------------------
def _as_points(points) -> Dict[int, Tuple[float, float]]:
    """Normaliza un dict de puntos (claves str o int) a {int: (x, y)} finitos."""
    out: Dict[int, Tuple[float, float]] = {}
    for key, value in dict(points or {}).items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        if value is None or len(value) != 2:
            continue
        x, y = float(value[0]), float(value[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        out[idx] = (x, y)
    return out


def _unit(v) -> np.ndarray:
    norm = float(np.hypot(v[0], v[1]))
    if norm < 1e-9:
        raise ValueError("Segmento de referencia degenerado")
    return np.asarray(v, dtype=float) / norm


def _rotate(v, deg: float) -> np.ndarray:
    t = math.radians(float(deg))
    c, s = math.cos(t), math.sin(t)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c], dtype=float)


def limb_distal_direction(points) -> np.ndarray:
    """Direccion distal aproximada del miembro: centro de rodilla (P1) -> tobillo (P2).

    Sirve para orientar el desplazamiento del corte y decidir que lado del plano
    es proximal y cual distal. Si faltan P1/P2 se asume "hacia abajo" (y crece).
    """
    pts = _as_points(points)
    if 1 in pts and 2 in pts:
        return _unit(np.array(pts[2], dtype=float) - np.array(pts[1], dtype=float))
    return np.array([0.0, 1.0], dtype=float)


def resection_direction(cut_type: str, distal_dir) -> np.ndarray:
    """Hacia donde avanza la reseccion segun el hueso.

    Tibia: el corte baja hacia distal (hacia el tobillo).
    Femur: el corte sube hacia proximal (hacia la cadera).
    """
    distal = np.array(distal_dir, dtype=float)
    if cut_type == "Femur":
        return -distal
    return distal


def _line_intersection(p1, p2, p3, p4):
    """Interseccion de dos rectas dadas por dos puntos cada una (o None)."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    return (
        (det1 * (x3 - x4) - (x1 - x2) * det2) / den,
        (det1 * (y3 - y4) - (y1 - y2) * det2) / den,
    )


# ---------------------------------------------------------------------------
# Geometria del corte
# ---------------------------------------------------------------------------
def reference_segment(points, cut_type: str, invert_ml: bool = False) -> dict:
    """Devuelve los puntos medial/lateral de la linea articular de referencia."""
    if cut_type not in CUT_TYPES:
        raise ValueError(f"cut_type debe ser uno de {sorted(CUT_TYPES)}")
    pts = _as_points(points)
    cfg = CUT_TYPES[cut_type]
    med_idx, lat_idx = cfg["med_idx"], cfg["lat_idx"]
    missing = [i for i in (med_idx, lat_idx) if i not in pts]
    if missing:
        raise ValueError(f"Faltan los puntos {missing} para el corte de {cut_type}")

    p_med, p_lat = pts[med_idx], pts[lat_idx]
    if invert_ml:
        p_med, p_lat = p_lat, p_med
        med_idx, lat_idx = lat_idx, med_idx

    width_px = float(math.hypot(p_lat[0] - p_med[0], p_lat[1] - p_med[1]))
    distal = limb_distal_direction(pts)

    # Cada hueso tiene su propia articulacion, su propio centro y su propio eje:
    #   Femur: centro = punto medio del condilo medial/lateral (P3-P4);
    #          eje mecanico femoral = centro femoral (P0) -> ese centro.
    #   Tibia: centro = punto medio del platillo medial/lateral (P5-P6);
    #          eje mecanico tibial  = ese centro -> centro del tobillo (P2).
    # No se comparte un unico "centro de rodilla" entre ambos huesos.
    p_med_v = np.array(p_med, dtype=float)
    p_lat_v = np.array(p_lat, dtype=float)
    center = (p_med_v + p_lat_v) / 2.0

    mech_dir = None
    p_axis = None
    if cut_type == "Femur" and 0 in pts:
        mech_dir = _unit(center - np.array(pts[0], dtype=float))
    elif cut_type == "Tibia" and 2 in pts:
        mech_dir = _unit(np.array(pts[2], dtype=float) - center)
    if mech_dir is not None:
        # El eje pasa por el centro articular por construccion; se deja el cruce
        # explicito para que el pivote sea siempre el punto sobre la linea.
        crossing = _line_intersection(p_med, p_lat, center, center + mech_dir)
        p_axis = (float(crossing[0]), float(crossing[1])) if crossing else (
            float(center[0]), float(center[1])
        )

    return {
        "cut_type": cut_type,
        "med_idx": med_idx,
        "lat_idx": lat_idx,
        "p_med": (float(p_med[0]), float(p_med[1])),
        "p_lat": (float(p_lat[0]), float(p_lat[1])),
        "width_px": width_px,
        "reference_label": cfg["reference"],
        "distal_dir": (float(distal[0]), float(distal[1])),
        "mech_dir": None if mech_dir is None else (float(mech_dir[0]), float(mech_dir[1])),
        "p_axis": p_axis,
    }


def plane_geometry(
    ref: dict,
    angle_deg: float,
    pivot_mode: str = "Mecanico",
    offset_px: float = 0.0,
) -> dict:
    """Plano de corte respecto del eje mecanico.

    Modelo por defecto ("Mecanico"): la profundidad se mide **sobre el eje
    mecanico** desde el punto donde ese eje cruza la linea articular, y 0 grados
    es el plano **perpendicular al eje mecanico**. Como la linea articular no es
    perpendicular a ese eje, a 0 grados cada condilo recibe una profundidad
    distinta: lo que se reseca depende de la anatomia.

    Modos "Medial"/"Lateral"/"Centro" conservan el modelo anterior (giro sobre la
    linea articular) y se mantienen para comparaciones.
    """
    if pivot_mode not in PIVOT_MODES:
        raise ValueError(f"pivot_mode debe ser uno de {PIVOT_MODES}")

    p_med = np.array(ref["p_med"], dtype=float)
    p_lat = np.array(ref["p_lat"], dtype=float)
    u = _unit(p_lat - p_med)  # medial -> lateral
    distal = np.array(ref.get("distal_dir", (0.0, 1.0)), dtype=float)

    mech = ref.get("mech_dir")
    p_axis = ref.get("p_axis")
    use_axis = pivot_mode == "Mecanico" and mech is not None and p_axis is not None

    if use_axis:
        m = _unit(np.array(mech, dtype=float))
        base_pivot = np.array(p_axis, dtype=float)
        # El avance va por el eje del hueso, hacia el hueso que se corta:
        # distal en la tibia, proximal en el femur.
        axis = m if ref["cut_type"] == "Tibia" else -m
        # Direccion del plano neutro (0 grados) = perpendicular al eje mecanico,
        # elegida del mismo sentido que medial->lateral.
        perp = np.array([-axis[1], axis[0]], dtype=float)
        if float(np.dot(perp, u)) < 0.0:
            perp = -perp
        base_dir = perp
    else:
        m = None
        axis = None
        base_pivot = {
            "Medial": p_med,
            "Lateral": p_lat,
            "Centro": (p_med + p_lat) / 2.0,
            "Mecanico": (p_med + p_lat) / 2.0,
        }[pivot_mode]
        base_dir = u

    # El giro se aplica con signo opuesto en femur y tibia para que un angulo
    # positivo siempre signifique lo mismo: valgo = mas reseccion lateral.
    spin = -1.0 if ref["cut_type"] == "Tibia" else 1.0
    res_dir = (axis if use_axis else
               resection_direction(ref["cut_type"], ref.get("distal_dir", (0.0, 1.0))))

    def _build(spin_sign: float, ang: float):
        direction_ = _rotate(base_dir, spin_sign * float(ang))
        normal_ = np.array([-direction_[1], direction_[0]], dtype=float)
        if float(np.dot(normal_, res_dir)) < 0.0:
            normal_ = -normal_
        if use_axis:
            # La profundidad avanza a lo largo del eje mecanico, no de la normal.
            pivot_ = base_pivot + axis * float(offset_px)
        else:
            pivot_ = base_pivot + normal_ * float(offset_px)
        # Profundidad con signo: positiva cuando el plano queda del lado del
        # hueso que se reseca; negativa si pasa al otro lado.
        depth_med_ = -float(np.dot(normal_, p_med - pivot_))
        depth_lat_ = -float(np.dot(normal_, p_lat - pivot_))
        return direction_, normal_, pivot_, depth_med_, depth_lat_

    direction, normal, pivot, depth_med, depth_lat = _build(spin, angle_deg)

    # Referencia a 0 grados: con el pivote mecanico la linea articular no es
    # perpendicular al eje, asi que ya hay una diferencia anatomica de base.
    _, _, _, depth_med0, depth_lat0 = _build(spin, 0.0)
    delta0 = depth_lat0 - depth_med0

    # La direccion medial->lateral se invierte entre pierna derecha e izquierda,
    # asi que el signo se corrige comparando con la referencia: un angulo positivo
    # debe resecar mas del lado lateral en cualquier lado y en cualquier hueso.
    if abs(float(angle_deg)) > 1e-9 and ((depth_lat - depth_med) - delta0) * float(angle_deg) < 0.0:
        direction, normal, pivot, depth_med, depth_lat = _build(-spin, angle_deg)

    proj_med = p_med + normal * depth_med
    proj_lat = p_lat + normal * depth_lat

    return {
        "pivot_mode": pivot_mode,
        "pivot": (float(pivot[0]), float(pivot[1])),
        "reference_direction": (float(u[0]), float(u[1])),
        "neutral_direction": (float(base_dir[0]), float(base_dir[1])),
        "p_axis": None if p_axis is None else (float(p_axis[0]), float(p_axis[1])),
        "mech_dir": None if mech is None else (float(mech[0]), float(mech[1])),
        "uses_axis": bool(use_axis),
        "direction": (float(direction[0]), float(direction[1])),
        "normal": (float(normal[0]), float(normal[1])),
        "distal_dir": (float(distal[0]), float(distal[1])),
        "resection_dir": (float(res_dir[0]), float(res_dir[1])),
        "offset_px": float(offset_px),
        "s_med_px": -depth_med,
        "s_lat_px": -depth_lat,
        "depth_med_px": depth_med,
        "depth_lat_px": depth_lat,
        "resection_med_px": max(0.0, depth_med),
        "resection_lat_px": max(0.0, depth_lat),
        "proj_med": (float(proj_med[0]), float(proj_med[1])),
        "proj_lat": (float(proj_lat[0]), float(proj_lat[1])),
    }


def simulate_cut(
    points,
    cut_type: str,
    angle_deg: float,
    pivot_mode: str = "Mecanico",
    width_mm: Optional[float] = None,
    invert_ml: bool = False,
    side: Optional[str] = None,
    offset_px: float = 0.0,
    offset_mm: Optional[float] = None,
    mm_per_px: Optional[float] = None,
    calibration_source: Optional[str] = None,
) -> dict:
    """Simula un corte y devuelve reseccion medial/lateral y la cuna resultante.

    Calibracion, en orden de prioridad:
      1. `mm_per_px` explicito (por ejemplo PixelSpacing de un DICOM).
      2. `width_mm` declarado por el usuario, dividido por el ancho articular en
         pixeles (aproximacion en pantallazos y PNG sin metadatos).
    Sin calibracion se informa solo en pixeles.

    La profundidad se puede dar en pixeles (`offset_px`) o en milimetros
    (`offset_mm`, requiere calibracion); `offset_mm` tiene prioridad.
    """
    ref = reference_segment(points, cut_type, invert_ml=invert_ml)

    if mm_per_px is not None and float(mm_per_px) > 0:
        scale = float(mm_per_px)
        source = calibration_source or "mm_per_px"
    elif width_mm and ref["width_px"] > 0:
        scale = float(width_mm) / ref["width_px"]
        source = calibration_source or "ancho articular declarado"
    else:
        scale = None
        source = None

    if offset_mm is not None and scale:
        offset_px_value = float(offset_mm) / scale
    else:
        offset_px_value = float(offset_px)

    geo = plane_geometry(ref, angle_deg, pivot_mode=pivot_mode, offset_px=offset_px_value)

    def to_mm(px_value):
        return None if scale is None else px_value * scale

    # Profundidades ya orientadas: positivo = mas reseccion en ese lado.
    delta_px = geo["resection_lat_px"] - geo["resection_med_px"]

    sim = {
        "side": side,
        "cut_type": ref["cut_type"],
        "angle_deg": float(angle_deg),
        "width_px": ref["width_px"],
        "width_mm": None if width_mm is None else float(width_mm),
        "mm_per_px": scale,
        "calibration_source": source,
        "offset_px": offset_px_value,
        "offset_mm": to_mm(offset_px_value),
        "reference_label": ref["reference_label"],
        "med_idx": ref["med_idx"],
        "lat_idx": ref["lat_idx"],
        "p_med": ref["p_med"],
        "p_lat": ref["p_lat"],
        "resection_med_px": geo["resection_med_px"],
        "resection_lat_px": geo["resection_lat_px"],
        "resection_med_mm": to_mm(geo["resection_med_px"]),
        "resection_lat_mm": to_mm(geo["resection_lat_px"]),
        "delta_px": delta_px,
        "delta_mm": to_mm(abs(delta_px)),
        "varus_valgus": ("Valgo" if angle_deg > 0.5 else ("Varo" if angle_deg < -0.5 else "Neutro")),
    }
    sim.update({k: v for k, v in geo.items() if k not in sim})
    return sim


def angle_for_delta_mm(delta_mm: float, width_mm: float) -> Optional[float]:
    """Angulo necesario para conseguir `delta_mm` de diferencia entre cortices.

    Inversa exacta de la relacion cuna = ancho * sin(angulo). Vale para cualquier
    pivote: el pivote solo reparte la profundidad, la diferencia entre cortices
    depende unicamente del angulo y del ancho.
    """
    if not width_mm or width_mm <= 0 or delta_mm is None:
        return None
    ratio = abs(float(delta_mm)) / float(width_mm)
    if ratio > 1.0:
        return None
    return math.degrees(math.asin(ratio))


# ---------------------------------------------------------------------------
# Carga de imagenes (DICOM, PNG/JPG, pantallazo)
# ---------------------------------------------------------------------------
def _mono_to_bgr(array: np.ndarray, photometric: str = "MONOCHROME2") -> np.ndarray:
    """Convierte una matriz monocroma a BGR uint8 con contraste estable."""
    arr = np.asarray(array, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Imagen sin datos validos")
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if hi - lo < 1e-6:
        lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-6:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if str(photometric).upper() == "MONOCHROME1":
        arr = 1.0 - arr
    gray = (arr * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def load_image(source, filename: Optional[str] = None) -> dict:
    """Carga una imagen desde ruta o bytes y detecta la calibracion disponible.

    Devuelve {'image': BGR, 'mm_per_px': float|None, 'source': ..., 'note': ...}.
    DICOM aporta PixelSpacing (milimetros reales); PNG/JPG/pantallazos no traen
    escala y quedan sin calibracion hasta que el usuario la declare.
    """
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        path = Path(source)
        raw = path.read_bytes()
        filename = filename or path.name

    name = filename or "imagen"
    suffix = Path(name).suffix.lower()
    looks_dicom = suffix in (".dcm", ".dicom") or raw[128:132] == b"DICM"

    if looks_dicom:
        try:
            import pydicom  # import local: el modulo funciona sin pydicom en PNG

            dataset = pydicom.dcmread(BytesIO(raw), force=True)
            array = dataset.pixel_array
            photometric = str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2"))
            slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)

            if array.ndim == 2:
                image = _mono_to_bgr(array.astype(np.float32) * slope + intercept, photometric)
            else:
                rgb = np.asarray(array[..., :3])
                if rgb.dtype != np.uint8:
                    rgb = cv2.normalize(rgb.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
                image = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

            spacing = getattr(dataset, "PixelSpacing", None)
            mm_per_px = None
            if spacing is not None and len(spacing) >= 2:
                mm_per_px = float((float(spacing[0]) + float(spacing[1])) / 2.0)

            note = "DICOM"
            if mm_per_px:
                note += f" · PixelSpacing {mm_per_px:.4f} mm/px"
            else:
                note += " · sin PixelSpacing"
            return {
                "image": image,
                "mm_per_px": mm_per_px,
                "source": "dicom",
                "note": note,
                "name": name,
            }
        except Exception as exc:  # noqa: BLE001 - se intenta como raster
            fallback_note = f"no se pudo leer como DICOM ({exc}); se intenta raster"
    else:
        fallback_note = ""

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo decodificar la imagen: {name}")
    note = "imagen raster · sin escala" if not fallback_note else fallback_note
    return {
        "image": image,
        "mm_per_px": None,
        "source": "raster",
        "note": note,
        "name": name,
    }


# ---------------------------------------------------------------------------
# Capas de hueso (proximal / distal / resecado)
# ---------------------------------------------------------------------------
def _roi_bounds(sim: dict, shape, factor: float = 2.4, full_strip: bool = False) -> Tuple[int, int, int, int]:
    """Region de trabajo alrededor de la articulacion.

    Con `full_strip=True` la ventana se extiende a lo largo de toda la extremidad
    (segun la direccion distal), para que el pintado de hueso no termine en un
    borde recto artificial.
    """
    height, width = shape[:2]
    xs = [sim["p_med"][0], sim["p_lat"][0], sim["proj_med"][0], sim["proj_lat"][0]]
    ys = [sim["p_med"][1], sim["p_lat"][1], sim["proj_med"][1], sim["proj_lat"][1]]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    half = max(sim["width_px"] * factor, 80.0)

    if not full_strip:
        x0 = int(max(0, cx - half))
        x1 = int(min(width, cx + half))
        y0 = int(max(0, cy - half))
        y1 = int(min(height, cy + half))
        return x0, y0, x1, y1

    distal = sim.get("distal_dir", (0.0, 1.0))
    if abs(float(distal[1])) >= abs(float(distal[0])):
        x0 = int(max(0, cx - half * 1.5))
        x1 = int(min(width, cx + half * 1.5))
        y0, y1 = 0, height
    else:
        y0 = int(max(0, cy - half * 1.5))
        y1 = int(min(height, cy + half * 1.5))
        x0, x1 = 0, width
    return x0, y0, x1, y1


def estimate_bone_mask(roi_bgr: np.ndarray, width_px: float) -> np.ndarray:
    """Mascara heuristica de hueso en la region de la rodilla.

    Dos umbrales de Otsu anidados: el primero separa aire de tejido; el segundo,
    dentro del tejido, separa tejido blando de hueso. Es una aproximacion por
    intensidad, no una segmentacion validada: sirve para visualizar, no para
    medir volumen oseo.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    tissue_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue = blur > tissue_thresh
    if int(np.count_nonzero(tissue)) < 500:
        return np.zeros_like(blur)

    tissue_values = blur[tissue].astype(np.uint8).reshape(-1, 1)
    bone_thresh, _ = cv2.threshold(
        tissue_values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Compromiso entre tejido blando y cortical: el punto medio reduce los huecos
    # del hueso esponjoso sin volver a incluir toda la partes blandas.
    cut = float(tissue_thresh) + 0.5 * (float(bone_thresh) - float(tissue_thresh))
    mask = np.where((blur > cut) & tissue, 255, 0).astype(np.uint8)

    size = int(max(3, round(width_px / 70.0))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def paint_bone_regions(
    canvas_bgr: np.ndarray,
    sim: dict,
    show_proximal: bool = True,
    show_distal: bool = True,
    show_resected: bool = True,
    alpha: float = 0.32,
) -> Tuple[np.ndarray, dict]:
    """Pinta hueso proximal, distal y resecado segun el plano de corte simulado.

    Devuelve la imagen pintada y las areas medidas en pixeles cuadrados (y mm2
    cuando hay calibracion).
    """
    canvas = np.ascontiguousarray(canvas_bgr).copy()
    stats: dict = {}
    x0, y0, x1, y1 = _roi_bounds(sim, canvas.shape, full_strip=True)
    roi = canvas[y0:y1, x0:x1]
    if roi.size == 0:
        return canvas, stats

    bone = estimate_bone_mask(roi, sim["width_px"])
    if not bone.any():
        return canvas, stats

    pivot = np.array(sim["pivot"], dtype=float)
    # `normal` apunta hacia donde avanza la reseccion (distal en tibia, proximal
    # en femur), asi "mas alla del plano" cambia de nombre segun el hueso.
    normal = np.array(sim["normal"], dtype=float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    beyond = (normal[0] * (xx - pivot[0]) + normal[1] * (yy - pivot[1])) > 0

    bone_bool = bone > 0
    if sim["cut_type"] == "Femur":
        proximal_mask = bone_bool & beyond
        distal_mask = bone_bool & ~beyond
    else:
        distal_mask = bone_bool & beyond
        proximal_mask = bone_bool & ~beyond

    resected_mask = np.zeros_like(bone_bool)
    if abs(math.sin(math.radians(sim["angle_deg"]))) > 1e-4 or abs(sim["offset_px"]) > 0.5:
        quad = np.array(
            [
                [int(round(sim["p_med"][0])) - x0, int(round(sim["p_med"][1])) - y0],
                [int(round(sim["p_lat"][0])) - x0, int(round(sim["p_lat"][1])) - y0],
                [int(round(sim["proj_lat"][0])) - x0, int(round(sim["proj_lat"][1])) - y0],
                [int(round(sim["proj_med"][0])) - x0, int(round(sim["proj_med"][1])) - y0],
            ],
            dtype=np.int32,
        )
        wedge = np.zeros_like(bone)
        cv2.fillConvexPoly(wedge, quad, 255, cv2.LINE_AA)
        resected_mask = bone_bool & (wedge > 0)

    tint = roi.copy()
    painted = False
    for enabled, mask, color in (
        (show_proximal, proximal_mask, PROXIMAL_COLOR),
        (show_distal, distal_mask, DISTAL_COLOR),
        (show_resected, resected_mask, RESECTED_COLOR),
    ):
        if enabled and mask.any():
            tint[mask] = color
            painted = True
    if painted:
        roi = cv2.addWeighted(tint, float(alpha), roi, 1.0 - float(alpha), 0)
        canvas[y0:y1, x0:x1] = roi

    scale = sim.get("mm_per_px")
    area_prox = int(np.count_nonzero(proximal_mask))
    area_dist = int(np.count_nonzero(distal_mask))
    area_res = int(np.count_nonzero(resected_mask))
    stats = {
        "proximal_px2": area_prox,
        "distal_px2": area_dist,
        "resected_px2": area_res,
    }
    if scale:
        factor = float(scale) ** 2
        stats.update(
            {
                "proximal_mm2": area_prox * factor,
                "distal_mm2": area_dist * factor,
                "resected_mm2": area_res * factor,
                "resected_cm2": area_res * factor / 100.0,
            }
        )
    return canvas, stats


# ---------------------------------------------------------------------------
# Dibujo del overlay (minimalista: una sola tinta + linea articular en gris)
# ---------------------------------------------------------------------------
ACCENT_COLOR = (226, 125, 48)    # azul institucional #307DE2 en BGR
JOINT_COLOR = (198, 198, 198)
HALO_COLOR = (24, 24, 24)


def _ipt(p) -> Tuple[int, int]:
    return (int(round(float(p[0]))), int(round(float(p[1]))))


def _dash_scale(width_px: float) -> int:
    return int(max(10.0, min(34.0, width_px * 0.022)))


def _dashed_line(canvas, p1, p2, color, thickness: int, dash: int = 18) -> None:
    a = np.array(p1, dtype=float)
    b = np.array(p2, dtype=float)
    total = float(np.linalg.norm(b - a))
    if total < 1e-6:
        return
    u = (b - a) / total
    step = 0.0
    while step < total:
        s0 = a + u * step
        s1 = a + u * min(step + dash, total)
        cv2.line(canvas, _ipt(s0), _ipt(s1), color, thickness, cv2.LINE_AA)
        step += dash * 2.4


def _put_label(canvas, text, origin, color, scale: float, thickness: int = 1,
               align: str = "left") -> None:
    x, y = _ipt(origin)
    if align == "right":
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x -= int(text_w)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, HALO_COLOR,
                thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _measurement(canvas, start, end, text, color, thickness: int, scale: float,
                 label_anchor=None, draw_label: bool = True,
                 align: str = "left") -> None:
    """Trazo fino de la distancia; la cifra va donde indique `label_anchor`."""
    p0, p1 = np.array(start, float), np.array(end, float)
    cv2.line(canvas, _ipt(p0), _ipt(p1), color, thickness, cv2.LINE_AA)
    if not draw_label:
        return
    if label_anchor is None:
        label_anchor = (p0 + p1) / 2.0 + np.array([0.0, -18.0])
    _put_label(canvas, text, label_anchor, color, scale, thickness, align=align)


def _soft_gray(canvas_bgr: np.ndarray, lighten: float = 0.30) -> np.ndarray:
    """Version aclarada en gris, para que las lineas dominen la imagen."""
    gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = gray * (1.0 - lighten) + 255.0 * lighten
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _bone_emphasis(canvas_bgr: np.ndarray, sim: dict, lighten: float = 0.58) -> np.ndarray:
    """Atenua partes blandas y mantiene el hueso a contraste pleno (figura limpia)."""
    x0, y0, x1, y1 = _roi_bounds(sim, canvas_bgr.shape, full_strip=True)
    out = _soft_gray(canvas_bgr, lighten=lighten)
    roi = canvas_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return out
    mask = estimate_bone_mask(roi, sim["width_px"])
    size = int(max(5, round(sim["width_px"] / 60.0))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    sub = out[y0:y1, x0:x1]
    sub[mask > 0] = roi[mask > 0]
    return out


def draw_cut_overlay(
    canvas_bgr: np.ndarray,
    sim: dict,
    line_thickness: int = 2,
    extend_px: Optional[int] = None,
    show_labels: bool = True,
    fill_wedge: bool = True,
    style: str = "line",
    show_values: bool = True,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Linea articular (gris), plano de corte (azul) y medidas.

    `style` controla el lenguaje visual, no la geometria:
      line    radiografia intacta + lineas finas (por defecto)
      clean   solo lineas, sin cifras ni velo
      blueprint  imagen aclarada en gris, trazo dominante
      diagram fondo blanco: solo geometria (figura esquematica)
    """
    if style not in ("line", "clean", "blueprint", "diagram"):
        raise ValueError(f"style debe ser line/clean/blueprint/diagram, no {style}")

    if style == "blueprint":
        base = _soft_gray(canvas_bgr, lighten=0.55)
    elif style == "diagram":
        base = np.full_like(canvas_bgr, 255)
    else:
        base = canvas_bgr

    if style == "clean":
        show_labels = False
        fill_wedge = False

    canvas = np.ascontiguousarray(base).copy()
    thickness = max(1, int(line_thickness))
    accent = ACCENT_COLOR if color is None else tuple(int(c) for c in color)
    width_px = float(canvas.shape[1])
    scale = max(0.45, min(1.05, width_px / 3000.0))

    if extend_px is None:
        extend_px = int(max(80.0, sim["width_px"] * 0.62))

    joint_center = np.array(
        [(sim["p_med"][0] + sim["p_lat"][0]) / 2.0,
         (sim["p_med"][1] + sim["p_lat"][1]) / 2.0],
        dtype=float,
    )

    # Cuna resecada: velo muy suave, solo para ubicarla.
    if fill_wedge and (abs(math.sin(math.radians(sim["angle_deg"]))) > 1e-4
                       or abs(sim["offset_px"]) > 0.5):
        quad = np.array(
            [sim["p_med"], sim["p_lat"], sim["proj_lat"], sim["proj_med"]],
            dtype=np.int32,
        )
        blend = canvas.copy()
        cv2.fillConvexPoly(blend, quad, accent, cv2.LINE_AA)
        canvas = cv2.addWeighted(blend, 0.12, canvas, 0.88, 0)

    # Linea articular de referencia, discreta.
    joint_color = (150, 150, 150) if style == "diagram" else JOINT_COLOR
    _dashed_line(canvas, sim["p_med"], sim["p_lat"], joint_color,
                 max(1, thickness), dash=_dash_scale(width_px))

    # Plano de corte simulado.
    pivot = np.array(sim["pivot"], dtype=float)
    direction = np.array(sim["direction"], dtype=float)
    start = pivot - direction * extend_px
    end = pivot + direction * extend_px
    cv2.line(canvas, _ipt(start), _ipt(end), accent, thickness + 1, cv2.LINE_AA)

    # Arco tenue del angulo (siempre el arco corto), medido desde el plano neutro.
    ref_dir = np.array(sim.get("neutral_direction") or sim["reference_direction"], dtype=float)
    if show_labels and abs(sim["angle_deg"]) > 0.2:
        radius = float(max(36.0, sim["width_px"] * 0.42))
        a_ref = math.degrees(math.atan2(ref_dir[1], ref_dir[0]))
        a_cut = math.degrees(math.atan2(direction[1], direction[0]))
        sweep = ((a_cut - a_ref + 180.0) % 360.0) - 180.0
        cv2.ellipse(canvas, _ipt(pivot), (int(radius), int(radius)), 0.0,
                    a_ref, a_ref + sweep, accent, max(1, thickness), cv2.LINE_AA)

    # Punto de giro: donde el eje mecanico cruza la linea articular.
    if sim.get("uses_axis"):
        r_mark = int(max(6.0, min(16.0, sim["width_px"] * 0.03)))
        cv2.circle(canvas, _ipt(pivot), r_mark, accent, max(1, thickness), cv2.LINE_AA)
        cv2.circle(canvas, _ipt(pivot), 2, accent, -1, cv2.LINE_AA)

    # Medidas: la cifra se coloca fuera del extremo correspondiente de la linea.
    u_ref = np.array(sim["reference_direction"], dtype=float)
    for value_key, proj_key, point_key, tag, side_sign in (
        ("resection_med_px", "proj_med", "p_med", "M", -1.0),
        ("resection_lat_px", "proj_lat", "p_lat", "L", 1.0),
    ):
        value_px = sim[value_key]
        if value_px < 0.5:
            continue
        if sim.get("mm_per_px"):
            text = f"{tag} {value_px * sim['mm_per_px']:.1f} mm"
        else:
            text = f"{tag} {value_px:.0f} px"
        anchor_point = np.array(sim[point_key], dtype=float)
        proj_point = np.array(sim[proj_key], dtype=float)
        label_anchor = (anchor_point + proj_point) / 2.0 + u_ref * (side_sign * sim["width_px"] * 0.30)
        # La cifra se alinea hacia fuera de la articulacion, nunca sobre la linea.
        align = "right" if label_anchor[0] < joint_center[0] else "left"
        _measurement(canvas, anchor_point, proj_point, text, accent,
                     max(1, thickness), scale,
                     label_anchor=label_anchor,
                     draw_label=show_labels and show_values, align=align)

    return canvas


def scale_sim(sim: dict, offset_x: float, offset_y: float, factor: float) -> dict:
    """Copia de `sim` expresada en coordenadas de una imagen recortada/escalada.

    Permite dibujar sobre el recorte ya reducido al tamano de presentacion, de
    modo que trazos, textos y marcas queden proporcionales a la imagen que se ve
    (y no a la resolucion original).
    """
    out = dict(sim)
    for key in ("p_med", "p_lat", "proj_med", "proj_lat", "pivot"):
        val = out.get(key)
        if val is not None:
            out[key] = ((float(val[0]) - offset_x) * factor,
                        (float(val[1]) - offset_y) * factor)
    val = out.get("p_axis")
    if val is not None:
        out["p_axis"] = ((float(val[0]) - offset_x) * factor,
                         (float(val[1]) - offset_y) * factor)
    out["width_px"] = float(sim["width_px"]) * factor
    out["offset_px"] = float(sim.get("offset_px") or 0.0) * factor
    for key in ("resection_med_px", "resection_lat_px", "depth_med_px", "depth_lat_px"):
        if key in out and out[key] is not None:
            out[key] = float(out[key]) * factor
    return out


def draw_mechanical_axes(
    canvas_bgr: np.ndarray,
    sims,
    color: Tuple[int, int, int] = (200, 200, 200),
    thickness: int = 1,
    extend_px: Optional[float] = None,
) -> np.ndarray:
    """Dibuja, en gris punteado tenue, el eje mecanico de cada hueso.

    Cada eje pasa por el punto donde cruza su linea articular y sigue la
    direccion mecano del hueso (femur: P0->centro femoral; tibia: centro
    tibial->P2). `sims` es una lista de simulaciones.
    """
    canvas = np.ascontiguousarray(canvas_bgr).copy()
    if isinstance(sims, dict):
        sims = [sims]
    dash = _dash_scale(canvas.shape[1])
    for sim in sims:
        mech = sim.get("mech_dir")
        p_axis = sim.get("p_axis")
        if not mech or not p_axis:
            continue
        d = _unit(np.array(mech, dtype=float))
        p = np.array(p_axis, dtype=float)
        ext = extend_px if extend_px is not None else max(120.0, sim["width_px"] * 1.7)
        _dashed_line(canvas, p - d * ext, p + d * ext, color,
                     max(1, thickness), dash=dash)
    return canvas


def draw_hud(canvas_bgr: np.ndarray, sim, badge: Optional[str] = None) -> np.ndarray:
    """Paneles translucidos con las cifras, dibujados encima de la imagen.

    `sim` puede ser un dict (un corte) o una lista de tuplas (titulo, sim, color)
    para mostrar varias osteotomias a la vez.
    """
    if isinstance(sim, dict):
        entries = [(sim.get("cut_type", "Corte"), sim, ACCENT_COLOR)]
    else:
        entries = [
            (e[0], e[1], ACCENT_COLOR if len(e) < 3 or e[2] is None else e[2])
            for e in sim
        ]

    canvas = np.ascontiguousarray(canvas_bgr).copy()
    height, width = canvas.shape[:2]
    # Escala del texto proporcional al tamano con que se muestra la imagen.
    scale = max(0.32, min(1.80, width / 1150.0))
    pad = int(16 * scale)
    line_h = int(30 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 0.62 * scale
    thick = max(1, int(round(1.4 * scale)))

    def rows_for(entry_sim):
        has_mm = entry_sim.get("mm_per_px") is not None
        unit = "mm" if has_mm else "px"

        def val(mm_key, px_key):
            v = entry_sim.get(mm_key) if has_mm else entry_sim.get(px_key)
            return "N/D" if v is None else f"{v:.1f} {unit}"

        return [
            ("Medial", val("resection_med_mm", "resection_med_px")),
            ("Lateral", val("resection_lat_mm", "resection_lat_px")),
            ("Cuna", val("delta_mm", "delta_px")),
            ("Angulo", f"{entry_sim['angle_deg']:+.1f} deg  ({entry_sim.get('varus_valgus', 'Neutro')})"),
        ]

    y_cursor = pad
    for title, entry_sim, color in entries:
        rows = rows_for(entry_sim)
        title_scale = f_scale * 0.92
        label_w = max(cv2.getTextSize(r[0], font, f_scale, thick)[0][0] for r in rows)
        value_w = max(cv2.getTextSize(r[1], font, f_scale, thick)[0][0] for r in rows)
        title_w = cv2.getTextSize(title, font, title_scale, thick)[0][0]
        panel_w = pad * 3 + max(label_w + value_w, title_w)
        panel_h = pad * 2 + int(line_h * 0.95) + line_h * len(rows)

        x0, y0 = pad, y_cursor
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      (255, 255, 255), -1, cv2.LINE_AA)
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      tuple(int(c) for c in color), max(1, thick), cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, 0.84, canvas, 0.16, 0)

        title_y = y0 + pad + int(line_h * 0.62)
        cv2.putText(canvas, title, (x0 + pad, title_y), font, title_scale,
                    tuple(int(c) for c in color), thick, cv2.LINE_AA)

        for i, (label, value) in enumerate(rows):
            base_y = y0 + pad + int(line_h * 0.95) + line_h * i + int(line_h * 0.72)
            cv2.putText(canvas, label, (x0 + pad, base_y), font, f_scale,
                        (120, 128, 140), thick, cv2.LINE_AA)
            (vw, _), _ = cv2.getTextSize(value, font, f_scale, thick)
            cv2.putText(canvas, value, (x0 + panel_w - pad - vw, base_y), font, f_scale,
                        (15, 23, 42), thick, cv2.LINE_AA)

        y_cursor += panel_h + int(pad * 0.75)

    if badge:
        bw, bh = cv2.getTextSize(badge, font, f_scale, thick)[0]
        bx1 = width - pad
        bx0 = bx1 - (bw + pad * 2)
        by0 = pad
        by1 = by0 + bh + pad * 2
        chip = canvas.copy()
        cv2.rectangle(chip, (bx0, by0), (bx1, by1), ACCENT_COLOR, -1, cv2.LINE_AA)
        canvas = cv2.addWeighted(chip, 0.90, canvas, 0.10, 0)
        cv2.putText(canvas, badge, (bx0 + pad, by1 - pad - int(bh * 0.25)), font,
                    f_scale, (255, 255, 255), thick, cv2.LINE_AA)

    return canvas


def summarize(sim: dict, bone_stats: Optional[dict] = None) -> list:
    """Resumen legible de la simulacion, listo para tabla."""
    has_mm = sim.get("mm_per_px") is not None
    unit = "mm" if has_mm else "px"
    rows = [
        ("Lado", sim.get("side") or "N/D"),
        ("Corte", sim["cut_type"]),
        ("Linea articular", sim["reference_label"]),
        ("Eje usado", "Mecanico del hueso" if sim.get("uses_axis") else sim["pivot_mode"]),
        ("Profundidad (offset)", f"{sim['offset_mm']:.1f} mm"
         if has_mm else f"{sim['offset_px']:.0f} px"),
        ("Angulo (grados)", f"{sim['angle_deg']:+.1f}"),
        ("Sentido", sim.get("varus_valgus", "Neutro")),
        ("Ancho articular (px)", f"{sim['width_px']:.1f}"),
    ]
    if has_mm:
        if sim.get("width_mm"):
            rows.append(("Ancho articular (mm)", f"{sim['width_mm']:.1f}"))
        rows.append(("Escala (mm/px)", f"{sim['mm_per_px']:.5f}"))
        if sim.get("calibration_source"):
            rows.append(("Calibracion", sim["calibration_source"]))
    rows.append((f"Reseccion medial ({unit})",
                 f"{sim['resection_med_mm']:.1f}" if has_mm else f"{sim['resection_med_px']:.1f}"))
    rows.append((f"Reseccion lateral ({unit})",
                 f"{sim['resection_lat_mm']:.1f}" if has_mm else f"{sim['resection_lat_px']:.1f}"))
    rows.append((f"Diferencia medial-lateral ({unit})",
                 f"{sim['delta_mm']:.1f}" if has_mm else f"{abs(sim['delta_px']):.1f}"))
    lado_mayor = "Lateral" if sim["delta_px"] > 0.5 else (
        "Medial" if sim["delta_px"] < -0.5 else "Simetrica")
    rows.append(("Reseccion mayor en", lado_mayor))
    if bone_stats:
        resected = bone_stats.get("resected_cm2")
        if resected is not None:
            rows.append(("Area resecada (aprox.)", f"{resected:.2f} cm2"))
        elif bone_stats.get("resected_px2") is not None:
            rows.append(("Area resecada (px2)", f"{bone_stats['resected_px2']:,}".replace(",", ".")))
    rows.append(("Advertencia", "Geometria 2D; requiere calibracion y validacion clinica"))
    return rows


def crop_bounds(sim: dict, shape, margin_factor: float = 1.6,
                margin_factor_y: Optional[float] = None):
    """Limites (x0, y0, x1, y1) del recorte alrededor de la articulacion."""
    height, width = shape[:2]
    factor_y = margin_factor if margin_factor_y is None else float(margin_factor_y)
    xs = [sim["p_med"][0], sim["p_lat"][0], sim["proj_med"][0], sim["proj_lat"][0]]
    ys = [sim["p_med"][1], sim["p_lat"][1], sim["proj_med"][1], sim["proj_lat"][1]]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    half_x = max(sim["width_px"] * float(margin_factor), 40.0)
    half_y = max(sim["width_px"] * factor_y, 40.0)
    return (
        int(max(0, cx - half_x)),
        int(max(0, cy - half_y)),
        int(min(width, cx + half_x)),
        int(min(height, cy + half_y)),
    )


def crop_around_joint(
    canvas_bgr: np.ndarray,
    sim: dict,
    margin_factor: float = 1.6,
    margin_factor_y: Optional[float] = None,
) -> np.ndarray:
    """Recorta la region de la rodilla simulada para inspeccion en detalle.

    `margin_factor_y` permite un encuadre panoramico (mas ancho que alto), que es
    el que mejor aprovecha la pantalla para una rodilla.
    """
    x0, y0, x1, y1 = crop_bounds(sim, canvas_bgr.shape, margin_factor, margin_factor_y)
    return canvas_bgr[y0:y1, x0:x1]
