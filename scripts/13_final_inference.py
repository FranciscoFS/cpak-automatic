import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DET_MODEL = BASE_DIR / "training_runs" / "telerx_yolo26s_768_b12-4" / "weights" / "best.pt"
DEFAULT_POSE_MODEL = BASE_DIR / "training_runs" / "telerx_pose_v2_ft" / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "visualizations" / "final_pipeline"

CLASS_INFO = {
    0: {"joint": "Cadera", "side": "Der"},
    1: {"joint": "Cadera", "side": "Izq"},
    2: {"joint": "Rodilla", "side": "Der"},
    3: {"joint": "Rodilla", "side": "Izq"},
    4: {"joint": "Tobillo", "side": "Der"},
    5: {"joint": "Tobillo", "side": "Izq"},
}

POSE_VISIBLE_INDICES = {
    "Cadera": [0],
    "Rodilla": [1, 3, 4, 5, 6, 7],
    "Tobillo": [2],
}

POINT_LABELS = {
    0: "Cabeza Femoral",
    1: "Centro Rodilla",
    2: "Centro Tobillo",
    3: "Condilo Medial",
    4: "Condilo Lateral",
    5: "Plateau Medial",
    6: "Plateau Lateral",
    7: "Notch Femoral",
}

SIDE_COLORS = {
    "Der": (60, 200, 60),
    "Izq": (0, 165, 255),
}

REQUIRED_METRIC_POINTS = [0, 1, 2, 3, 4, 5, 6, 7]
AHKA_FORMULA = "MPTA_MINUS_LDFA"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline final CPAK: deteccion, pose, overlay y calculo clinico sobre una TeleRx completa."
    )
    parser.add_argument("--source", required=True, help="Ruta a una imagen o a un directorio con imagenes.")
    parser.add_argument("--det-model", default=str(DEFAULT_DET_MODEL), help="Ruta al modelo de deteccion.")
    parser.add_argument("--pose-model", default=str(DEFAULT_POSE_MODEL), help="Ruta al modelo de pose.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directorio para overlays y JSON.")
    parser.add_argument("--det-conf", type=float, default=0.20, help="Confianza minima para deteccion de zonas.")
    parser.add_argument("--pose-conf", type=float, default=0.20, help="Confianza minima para pose sobre cada crop.")
    parser.add_argument("--padding", type=float, default=0.15, help="Padding extra alrededor de cada bounding box.")
    parser.add_argument("--json-only", action="store_true", help="Guardar solo JSON y omitir overlays.")
    return parser.parse_args()


def collect_images(source: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted([p for p in source.iterdir() if p.is_file() and p.suffix.lower() in exts])
    raise FileNotFoundError(f"No existe la ruta de entrada: {source}")


def validate_model_path(model_path: Path, label: str):
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo de {label}: {model_path}")


def pick_best_detections(result):
    best = {}
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        if cls_id not in best or conf > best[cls_id]["confidence"]:
            best[cls_id] = {
                "class_id": cls_id,
                "confidence": conf,
                "xyxy": xyxy,
                "joint": CLASS_INFO.get(cls_id, {}).get("joint", f"Clase_{cls_id}"),
                "side": CLASS_INFO.get(cls_id, {}).get("side", "NA"),
            }
    return best


def bbox_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def validate_side_coherence(detections):
    """
    Valida y corrige asignaciones Der/Izq usando posición X espacial.

    Convención del dataset: Der tiene menor X (lado izquierdo de la imagen),
    Izq tiene mayor X (lado derecho de la imagen), convención radiológica estándar.

    Estrategia:
    1. Buscar "anclas": articulaciones donde se detectaron AMBOS lados.
       Estas dan la referencia real de Der_x vs Izq_x en esta imagen.
    2. Calcular el punto medio entre Der y Izq usando las anclas.
    3. Verificar CADA detección: si su X está claramente del lado equivocado
       (más allá de un margen del 10% del span), corregir el lado.

    Si no hay anclas (ej: solo se detectó un lado), retorna sin cambios.
    """
    if not detections:
        return detections

    # Paso 0: ordenar por posición dentro de cada articulación (izquierda->Der, derecha->Izq).
    by_joint_keys = {}
    for key, det in detections.items():
        by_joint_keys.setdefault(det["joint"], []).append(key)

    for joint, keys in by_joint_keys.items():
        if len(keys) < 2:
            continue
        keys_sorted = sorted(keys, key=lambda k: (detections[k]["xyxy"][0] + detections[k]["xyxy"][2]) / 2)
        left_key = keys_sorted[0]
        right_key = keys_sorted[-1]
        if detections[left_key]["side"] != "Der":
            print(f"  [Orden-X] {joint} {detections[left_key]['side']}→Der")
            detections[left_key]["side"] = "Der"
        if detections[right_key]["side"] != "Izq":
            print(f"  [Orden-X] {joint} {detections[right_key]['side']}→Izq")
            detections[right_key]["side"] = "Izq"

    # Paso 1: calcular anclas globales (preferencia Cadera/Tobillo por estabilidad).
    preferred_joints = ["Cadera", "Tobillo"]

    def collect_anchors(joints_subset=None):
        der_xs, izq_xs = [], []
        for joint, keys in by_joint_keys.items():
            if joints_subset is not None and joint not in joints_subset:
                continue
            side_map = {}
            for key in keys:
                det = detections.get(key)
                if det is None:
                    continue
                if det["side"] not in ("Der", "Izq"):
                    continue
                xc = (det["xyxy"][0] + det["xyxy"][2]) / 2
                side_map[det["side"]] = xc
            if "Der" in side_map and "Izq" in side_map:
                der_xs.append(side_map["Der"])
                izq_xs.append(side_map["Izq"])
        return der_xs, izq_xs

    der_anchor_xs, izq_anchor_xs = collect_anchors(preferred_joints)
    if not der_anchor_xs:
        der_anchor_xs, izq_anchor_xs = collect_anchors()
    if not der_anchor_xs:
        return detections

    der_x_ref = float(np.median(der_anchor_xs))
    izq_x_ref = float(np.median(izq_anchor_xs))
    midpoint = (der_x_ref + izq_x_ref) / 2
    span = abs(der_x_ref - izq_x_ref)
    margin = max(8.0, span * 0.10)

    # Paso 2: coherencia global por eje X contra midpoint.
    for key, det in detections.items():
        if det["side"] not in ("Der", "Izq"):
            continue

        xc = (det["xyxy"][0] + det["xyxy"][2]) / 2
        expected = "Der" if xc < midpoint else "Izq"
        if expected != det["side"] and abs(xc - midpoint) > margin:
            old_side = det["side"]
            det["side"] = expected
            print(
                f"  [Coherencia-X] {det['joint']} cls{det['class_id']} "
                f"{old_side}→{expected} (x={xc:.0f}, mid={midpoint:.0f})"
            )

    # Paso 3: resolver duplicados por superposición extrema en la misma mitad.
    to_delete = set()
    for joint, keys in by_joint_keys.items():
        active = [k for k in keys if k in detections]
        if len(active) < 2:
            continue
        a, b = active[0], active[1]
        det_a = detections[a]
        det_b = detections[b]
        xa = (det_a["xyxy"][0] + det_a["xyxy"][2]) / 2
        xb = (det_b["xyxy"][0] + det_b["xyxy"][2]) / 2
        expected_a = "Der" if xa < midpoint else "Izq"
        expected_b = "Der" if xb < midpoint else "Izq"
        iou = bbox_iou_xyxy(det_a["xyxy"], det_b["xyxy"])
        if iou > 0.70 and expected_a == expected_b:
            drop_key, keep_key = (a, b) if det_a["confidence"] <= det_b["confidence"] else (b, a)
            to_delete.add(drop_key)
            detections[keep_key]["side"] = expected_a
            print(
                f"  [Conflicto-IoU] {joint} IoU={iou:.2f} ambos->{expected_a}; "
                f"mantener cls{detections[keep_key]['class_id']} descartar cls{detections[drop_key]['class_id']}"
            )

    for key in to_delete:
        detections.pop(key, None)

    # Paso 4: garantizar una sola detección por (joint, side), quedándose con mayor confianza.
    best_by_joint_side = {}
    for key, det in detections.items():
        joint_side = (det["joint"], det["side"])
        best_key = best_by_joint_side.get(joint_side)
        if best_key is None or det["confidence"] > detections[best_key]["confidence"]:
            best_by_joint_side[joint_side] = key

    keep_keys = set(best_by_joint_side.values())
    for key in list(detections.keys()):
        if key not in keep_keys:
            detections.pop(key, None)

    return detections


def reflect_missing_detections(detections: dict, image_width: int):
    """
    Refleja bboxes contralaterales si falta detección en un lado.
    
    Si Der tiene Rodilla pero Izq no, espeja el bbox de Rodilla_Der → Rodilla_Izq.
    
    Args:
        detections: dict de cls_id → {confidence, xyxy, joint, side, ...}
        image_width: ancho de la imagen para calcular la reflexión
    
    Returns:
        detections con detecciones reflejadas añadidas (si es posible)
    """
    if not detections:
        return detections
    
    joints_by_side = {"Der": set(), "Izq": set()}
    for det in detections.values():
        side = det["side"]
        joint = det["joint"]
        if side in joints_by_side:
            joints_by_side[side].add(joint)
    
    # Identificar qué articulaciones faltan en cada lado
    all_joints = joints_by_side["Der"] | joints_by_side["Izq"]
    missing_in_izq = all_joints - joints_by_side["Izq"]
    missing_in_der = all_joints - joints_by_side["Der"]
    
    reflected = {}
    next_cls_id = max(detections.keys()) + 1 if detections else 0
    
    # Reflejar articulaciones faltantes en Izq desde Der
    for joint in missing_in_izq:
        for det in detections.values():
            if det["joint"] == joint and det["side"] == "Der":
                x1, y1, x2, y2 = det["xyxy"]
                # Reflexión horizontal: x' = width - x
                x1_reflected = image_width - x2
                x2_reflected = image_width - x1
                
                reflected[next_cls_id] = {
                    "class_id": next_cls_id,
                    "confidence": det["confidence"] * 0.95,  # Penalizar ligeramente
                    "xyxy": [x1_reflected, y1, x2_reflected, y2],
                    "joint": joint,
                    "side": "Izq",
                    "reflected": True,
                }
                print(f"  [Reflexión] {joint} Der → Izq (clase {next_cls_id}, conf {reflected[next_cls_id]['confidence']:.3f})")
                next_cls_id += 1
                break
    
    # Reflejar articulaciones faltantes en Der desde Izq
    for joint in missing_in_der:
        for det in detections.values():
            if det["joint"] == joint and det["side"] == "Izq":
                x1, y1, x2, y2 = det["xyxy"]
                # Reflexión horizontal
                x1_reflected = image_width - x2
                x2_reflected = image_width - x1
                
                reflected[next_cls_id] = {
                    "class_id": next_cls_id,
                    "confidence": det["confidence"] * 0.95,
                    "xyxy": [x1_reflected, y1, x2_reflected, y2],
                    "joint": joint,
                    "side": "Der",
                    "reflected": True,
                }
                print(f"  [Reflexión] {joint} Izq → Der (clase {next_cls_id}, conf {reflected[next_cls_id]['confidence']:.3f})")
                next_cls_id += 1
                break
    
    detections.update(reflected)
    return detections


def padded_box(xyxy, image_shape, padding):
    """
    Expande un bounding box un porcentaje de su tamaño.
    
    El padding se aplica al bounding box de DETECCIÓN (no al modelo de pose).
    Sirve para darle más contexto anatómico al modelo de pose.
    
    Ejemplo: padding=0.15 y BB de 100x100 px → BB expandido a ~130x130 px
    
    Args:
        xyxy: coordenadas originales del BB (x1, y1, x2, y2)
        image_shape: (height, width) de la imagen
        padding: fracción decimal (0.15 = 15%)
    
    Returns:
        Coordenadas expandidas (nx1, ny1, nx2, ny2) limitadas a los bordes de la imagen
    """
    height, width = image_shape[:2]
    x1, y1, x2, y2 = xyxy
    bw = x2 - x1
    bh = y2 - y1
    pad_w = bw * padding
    pad_h = bh * padding
    nx1 = max(0, int(round(x1 - pad_w)))
    ny1 = max(0, int(round(y1 - pad_h)))
    nx2 = min(width, int(round(x2 + pad_w)))
    ny2 = min(height, int(round(y2 + pad_h)))
    return nx1, ny1, nx2, ny2


def infer_crop_keypoints(pose_model: YOLO, crop, joint_name: str, pose_conf: float):
    result = pose_model.predict(source=crop, conf=pose_conf, verbose=False)[0]
    if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return {}

    pred_xy = result.keypoints.xy[0].cpu().numpy()
    points = {}
    for idx in POSE_VISIBLE_INDICES[joint_name]:
        px = float(pred_xy[idx][0])
        py = float(pred_xy[idx][1])
        if px > 1 or py > 1:
            points[idx] = (px, py)
    return points


def build_side_points(image, side_detections, pose_model: YOLO, pose_conf: float, padding: float):
    """
    Procesa cada zona (Cadera, Rodilla, Tobillo) de un lado:
    
    1. Toma el BB de detección
    2. Lo expande por el porcentaje de padding (más contexto para pose)
    3. Extrae crop expandido
    4. Corre YOLO Pose en el crop expandido
    5. Mapea coordenadas locales del crop → globales de imagen
    
    El padding NO afecta cómo funciona el modelo de pose.
    Solo controla cuánto contexto ve el modelo alrededor de cada zona.
    """
    global_points = {}
    crops = {}
    for joint_name in ["Cadera", "Rodilla", "Tobillo"]:
        detection = side_detections.get(joint_name)
        if detection is None:
            continue

        x1, y1, x2, y2 = padded_box(detection["xyxy"], image.shape, padding)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        local_points = infer_crop_keypoints(pose_model, crop, joint_name, pose_conf)
        mapped_points = {}
        for idx, (px, py) in local_points.items():
            mapped_points[idx] = (x1 + px, y1 + py)
            global_points[idx] = mapped_points[idx]

        crops[joint_name] = {
            "crop_box": [x1, y1, x2, y2],
            "keypoints": {str(idx): [round(pt[0], 2), round(pt[1], 2)] for idx, pt in mapped_points.items()},
        }

    return global_points, crops


def vector(p1, p2):
    return np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)


def line_angle_deg(p1, p2, p3, p4):
    v1 = vector(p1, p2)
    v2 = vector(p3, p4)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return None

    cosine = float(np.dot(v1, v2) / (norm1 * norm2))
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.degrees(math.acos(cosine))
    return min(angle, 180.0 - angle)


def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-8:
        return None

    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    px = (det1 * (x3 - x4) - (x1 - x2) * det2) / den
    py = (det1 * (y3 - y4) - (y1 - y2) * det2) / den
    return (float(px), float(py))


def angle_at_vertex_deg(vertex, point_a, point_b):
    va = vector(vertex, point_a)
    vb = vector(vertex, point_b)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return None

    cosine = float(np.dot(va, vb) / (norm_a * norm_b))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def perpendicular_distance_point_to_line(point, line_p1, line_p2):
    """
    Calcula la distancia perpendicular desde un punto a una línea definida por dos puntos.
    
    Fórmula: d = ||(p - p1) × (p2 - p1)|| / ||p2 - p1||
    
    Args:
        point: (x, y) punto
        line_p1: (x1, y1) primer punto de la línea
        line_p2: (x2, y2) segundo punto de la línea
    
    Returns:
        Distancia perpendicular (en píxeles, requiere pixel_spacing del DICOM para mm)
    """
    px, py = float(point[0]), float(point[1])
    x1, y1 = float(line_p1[0]), float(line_p1[1])
    x2, y2 = float(line_p2[0]), float(line_p2[1])
    
    # Vector de la línea
    dx = x2 - x1
    dy = y2 - y1
    line_len_sq = dx * dx + dy * dy
    
    if line_len_sq < 1e-10:
        # Línea degenerada (puntos muy cercanos)
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    
    # Proyección del punto sobre la línea
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / line_len_sq))
    
    # Punto proyectado
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    # Distancia perpendicular
    dist = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
    return dist


def classify_ahka(value):
    if value < -2.0:
        return "Varo"
    if value <= 2.0:
        return "Neutro"
    return "Valgo"


def classify_jlo(value):
    if value < 177.0:
        return "Apex Distal"
    if value <= 181.0:
        return "Neutro"
    return "Apex Proximal"


# CPAK type I–IX: rows = aHKA (Varo/Neutro/Valgo), cols = JLO (Distal/Neutro/Proximal)
_CPAK_TYPE_GRID = {
    ("Varo",   "Apex Distal"):   "I",
    ("Neutro", "Apex Distal"):   "II",
    ("Valgo",  "Apex Distal"):   "III",
    ("Varo",   "Neutro"):        "IV",
    ("Neutro", "Neutro"):        "V",
    ("Valgo",  "Neutro"):        "VI",
    ("Varo",   "Apex Proximal"): "VII",
    ("Neutro", "Apex Proximal"): "VIII",
    ("Valgo",  "Apex Proximal"): "IX",
}


def classify_cpak_type(ahka_class: str, jlo_class: str) -> str:
    return _CPAK_TYPE_GRID.get((ahka_class, jlo_class), "?")


def calculate_metrics(points):
    missing = [idx for idx in REQUIRED_METRIC_POINTS if idx not in points]
    if missing:
        return {"status": "incomplete", "missing_points": missing}

    fem_intersection = line_intersection(points[0], points[7], points[3], points[4])
    tib_intersection = line_intersection(points[1], points[2], points[5], points[6])
    if fem_intersection is None or tib_intersection is None:
        return {"status": "invalid_geometry"}

    # LDFA ambiguity is common (internal vs external). We report both and select one explicitly.
    ldfa_external = angle_at_vertex_deg(fem_intersection, points[0], points[4])
    # MPTA: angle at tibial/plateau intersection using distal tibial axis ray and medial plateau ray.
    mpta = angle_at_vertex_deg(tib_intersection, points[2], points[5])
    if ldfa_external is None or mpta is None:
        return {"status": "invalid_geometry"}

    ldfa_internal = 180.0 - ldfa_external
    ldfa = ldfa_external

    ahka = mpta - ldfa
    jlo = mpta + ldfa

    # HKA angular real (no aritmético): ángulo entre eje femoral P0->P7 y eje tibial P1->P2
    hka = line_angle_deg(points[0], points[7], points[1], points[2])
    if hka is None:
        return {"status": "invalid_geometry"}
    # Asignar signo según aHKA: negativo = varo, positivo = valgo
    if ahka < 0:
        hka = -hka
    
    # Línea de Mikulicz: desde P0 (Cabeza Femoral) a P2 (Centro Tobillo)
    p0 = points[0]  # Cabeza Femoral
    p2 = points[2]  # Centro Tobillo
    p1 = points[1]  # Centro Rodilla (para MAD)
    
    # MAD: distancia perpendicular desde P1 a la línea Mikulicz (P0-P2)
    mad_px = perpendicular_distance_point_to_line(p1, p0, p2)
    
    return {
        "status": "ok",
        "LDFA": round(ldfa, 3),
        "LDFA_external": round(ldfa_external, 3),
        "LDFA_internal": round(ldfa_internal, 3),
        "LDFA_mode": "EXTERNAL_FIXED",
        "MPTA": round(mpta, 3),
        "HKA": round(hka, 3),
        "HKA_definition": "ANGULO_ENTRE_EJES_CON_SIGNO_NEGATIVO_VARO_POSITIVO_VALGO",
        "aHKA": round(ahka, 3),
        "aHKA_formula": AHKA_FORMULA,
        "JLO": round(jlo, 3),
        "aHKA_class": classify_ahka(ahka),
        "JLO_class": classify_jlo(jlo),
        "CPAK": f"{classify_ahka(ahka)} | {classify_jlo(jlo)}",
        "CPAK_type": classify_cpak_type(classify_ahka(ahka), classify_jlo(jlo)),
        "MAD_px": round(mad_px, 2),
        "MAD_note": "Distancia perpendicular (píxeles) desde centro rodilla a línea Mikulicz. Requiere pixel_spacing del DICOM para convertir a mm.",
    }


def get_overlay_style(canvas):
    h, w = canvas.shape[:2]
    base = max(600, min(h, w))
    return {
        "box_thickness": max(2, int(round(base / 650))),
        "line_thickness": max(1, int(round(base / 900))),
        "point_radius": max(4, int(round(base / 180))),
        "font_main": max(0.45, base / 2200.0),
        "font_small": max(0.4, base / 2600.0),
        "text_thickness": max(1, int(round(base / 1200))),
        "margin": max(10, int(round(base / 120))),
        "label_pad": max(5, int(round(base / 220))),
    }


def clip_text_to_width(text: str, max_width: int, font_scale: float, thickness: int):
    if max_width <= 0:
        return ""
    current = text
    while current:
        text_width = cv2.getTextSize(current, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        if text_width <= max_width:
            return current
        if len(current) <= 4:
            return "..."
        current = current[:-4] + "..."
    return ""


def project_line_to_image(p1, p2, width: int, height: int):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    dy = y2 - y1

    candidates = []

    if abs(dx) > 1e-8:
        t_left = (0.0 - x1) / dx
        y_left = y1 + t_left * dy
        if 0.0 <= y_left <= (height - 1):
            candidates.append((0.0, y_left))

        t_right = ((width - 1) - x1) / dx
        y_right = y1 + t_right * dy
        if 0.0 <= y_right <= (height - 1):
            candidates.append((float(width - 1), y_right))

    if abs(dy) > 1e-8:
        t_top = (0.0 - y1) / dy
        x_top = x1 + t_top * dx
        if 0.0 <= x_top <= (width - 1):
            candidates.append((x_top, 0.0))

        t_bottom = ((height - 1) - y1) / dy
        x_bottom = x1 + t_bottom * dx
        if 0.0 <= x_bottom <= (width - 1):
            candidates.append((x_bottom, float(height - 1)))

    unique = []
    for pt in candidates:
        if not any(abs(pt[0] - q[0]) < 1e-6 and abs(pt[1] - q[1]) < 1e-6 for q in unique):
            unique.append(pt)

    if len(unique) < 2:
        return None

    best_pair = None
    best_dist = -1.0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            dist = (unique[i][0] - unique[j][0]) ** 2 + (unique[i][1] - unique[j][1]) ** 2
            if dist > best_dist:
                best_dist = dist
                best_pair = (unique[i], unique[j])

    if best_pair is None:
        return None

    p_start = (int(round(best_pair[0][0])), int(round(best_pair[0][1])))
    p_end = (int(round(best_pair[1][0])), int(round(best_pair[1][1])))
    return p_start, p_end


def draw_simple_line(canvas, p1, p2, color, thickness):
    """Dibuja una línea simple entre dos puntos sin extender."""
    pt1 = (int(round(p1[0])), int(round(p1[1])))
    pt2 = (int(round(p2[0])), int(round(p2[1])))
    cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)


def draw_projected_line(canvas, p1, p2, color, thickness, line_style: str = "solid"):
    """
    Dibuja una línea proyectada desde p1 a p2 hasta el borde de la imagen.
    
    Args:
        line_style: "solid" o "dashed"
    """
    projected = project_line_to_image(p1, p2, canvas.shape[1], canvas.shape[0])
    if projected is None:
        return
    
    if line_style == "dashed":
        # Dibujar línea punteada
        pt1, pt2 = projected[0], projected[1]
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1:
            return
        
        segments = int(length / 20)  # Segmentos de ~20 px
        for i in range(0, segments, 2):
            x1 = int(pt1[0] + (dx / segments) * i)
            y1 = int(pt1[1] + (dy / segments) * i)
            x2 = int(pt1[0] + (dx / segments) * (i + 1))
            y2 = int(pt1[1] + (dy / segments) * (i + 1))
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    else:
        cv2.line(canvas, projected[0], projected[1], color, thickness, cv2.LINE_AA)



def draw_extended_line(canvas, p1, p2, color, thickness, extend: float = 0.20):
    """Dibuja una línea entre p1 y p2 extendida `extend` fracción más allá de cada extremo."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    dy = y2 - y1
    ex1 = (int(round(x1 - dx * extend)), int(round(y1 - dy * extend)))
    ex2 = (int(round(x2 + dx * extend)), int(round(y2 + dy * extend)))
    h, w = canvas.shape[:2]
    ex1 = (max(0, min(w - 1, ex1[0])), max(0, min(h - 1, ex1[1])))
    ex2 = (max(0, min(w - 1, ex2[0])), max(0, min(h - 1, ex2[1])))
    cv2.line(canvas, ex1, ex2, color, thickness, cv2.LINE_AA)


def draw_detection_boxes(canvas, detections, style):
    for det in detections.values():
        side = det["side"]
        color = SIDE_COLORS.get(side, (255, 255, 255))
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        label = f"{det['joint']}_{side} {det['confidence']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, style["box_thickness"])
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, style["font_small"], style["text_thickness"])[0]
        pad = style["label_pad"]
        tag_h = text_size[1] + (pad * 2)
        tag_w = text_size[0] + (pad * 2)
        tag_y1 = max(0, y1 - tag_h)
        tag_y2 = y1
        tag_x2 = min(canvas.shape[1], x1 + tag_w)
        cv2.rectangle(canvas, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + pad, tag_y2 - pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            style["font_small"],
            (20, 20, 20),
            style["text_thickness"],
            cv2.LINE_AA,
        )


def draw_side_overlay(canvas, side: str, points, metrics, style, draw_axes: bool = True, draw_points: bool = True, draw_mikulicz: bool = False):
    color = SIDE_COLORS.get(side, (255, 255, 255))
    point_radius = style["point_radius"]

    if draw_axes:
        # Ejes: solo de punto a punto (sin extender)
        for start, end in [(0, 7), (1, 2)]:
            if start in points and end in points:
                draw_simple_line(canvas, points[start], points[end], color, style["line_thickness"])
        # Cóndilos y platillos: solo 20% más allá de cada punto (rodilla)
        for start, end in [(3, 4), (5, 6)]:
            if start in points and end in points:
                draw_extended_line(canvas, points[start], points[end], color, style["line_thickness"], extend=0.20)

    # Línea de Mikulicz (P0 Cadera a P2 Tobillo) - punteada, de punto a punto
    if draw_mikulicz and 0 in points and 2 in points:
        mikulicz_color = (128, 128, 255)  # Azul pálido
        # Dibuja la línea punteada manualmente punto a punto
        pt1 = (int(round(points[0][0])), int(round(points[0][1])))
        pt2 = (int(round(points[2][0])), int(round(points[2][1])))
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length > 1:
            segments = int(length / 20)  # Segmentos de ~20 px
            for i in range(0, segments, 2):
                x1 = int(pt1[0] + (dx / segments) * i)
                y1 = int(pt1[1] + (dy / segments) * i)
                x2 = int(pt1[0] + (dx / segments) * (i + 1))
                y2 = int(pt1[1] + (dy / segments) * (i + 1))
                cv2.line(canvas, (x1, y1), (x2, y2), mikulicz_color, style["line_thickness"], cv2.LINE_AA)

    if draw_points:
        for idx, (px, py) in points.items():
            center = (int(round(px)), int(round(py)))
            cv2.circle(canvas, center, point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, point_radius + style["line_thickness"], (255, 255, 255), style["line_thickness"], cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"P{idx}",
                (center[0] + style["label_pad"], center[1] - style["label_pad"]),
                cv2.FONT_HERSHEY_SIMPLEX,
                style["font_small"],
                color,
                style["text_thickness"],
                cv2.LINE_AA,
            )




def render_overlay_from_payload(
    image_path: Path,
    payload: dict,
    draw_boxes: bool = True,
    draw_axes: bool = True,
    draw_mikulicz: bool = False,
    line_thickness_override: int = None,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"No se pudo leer la imagen para render: {image_path}")

    canvas = image.copy()
    style = get_overlay_style(canvas)
    if line_thickness_override is not None:
        style["line_thickness"] = max(1, int(line_thickness_override))

    detections = {}
    for cls_id_str, det in payload.get("detections", {}).items():
        detections[int(cls_id_str)] = {
            "joint": det["joint"],
            "side": det["side"],
            "confidence": float(det["confidence"]),
            "xyxy": [float(v) for v in det["xyxy"]],
        }

    if draw_boxes:
        draw_detection_boxes(canvas, detections, style)

    for side_name in ["Der", "Izq"]:
        side_info = payload.get("sides", {}).get(side_name, {})
        points = {
            int(idx): (float(vals[0]), float(vals[1]))
            for idx, vals in side_info.get("points", {}).items()
        }
        metrics = side_info.get("metrics", {})
        draw_side_overlay(canvas, side_name, points, metrics, style, draw_axes=draw_axes, draw_points=True, draw_mikulicz=draw_mikulicz)

    return canvas


def process_image(
    det_model: YOLO,
    pose_model: YOLO,
    image_path: Path,
    output_dir: Path,
    det_conf: float,
    pose_conf: float,
    padding: float,
    json_only: bool = False,
    draw_boxes: bool = True,
    draw_axes: bool = True,
    draw_mikulicz: bool = False,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"No se pudo leer la imagen: {image_path}")

    det_result = det_model.predict(source=str(image_path), conf=det_conf, verbose=False)[0]
    detections = pick_best_detections(det_result)
    detections = validate_side_coherence(detections)
    
    # Reflexionar bboxes contralaterales si faltan articulaciones
    image_width = image.shape[1]
    detections = reflect_missing_detections(detections, image_width)

    sides = {"Der": {}, "Izq": {}}
    for det in detections.values():
        if det["side"] in sides:
            sides[det["side"]][det["joint"]] = det

    result_payload = {"image": str(image_path), "detections": {}, "sides": {}}
    for cls_id, det in detections.items():
        result_payload["detections"][str(cls_id)] = {
            "joint": det["joint"],
            "side": det["side"],
            "confidence": round(det["confidence"], 4),
            "xyxy": [round(v, 2) for v in det["xyxy"]],
            "reflected": det.get("reflected", False),
        }

    for side_name, side_detections in sides.items():
        points, crops = build_side_points(image, side_detections, pose_model, pose_conf, padding)
        metrics = calculate_metrics(points)
        result_payload["sides"][side_name] = {
            "detections_found": sorted(side_detections.keys()),
            "points": {str(idx): [round(pt[0], 2), round(pt[1], 2)] for idx, pt in points.items()},
            "crops": crops,
            "metrics": metrics,
        }

    canvas = None
    if not json_only:
        canvas = render_overlay_from_payload(
            image_path,
            result_payload,
            draw_boxes=draw_boxes,
            draw_axes=draw_axes,
            draw_mikulicz=draw_mikulicz,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{image_path.stem}_final_metrics.json"
    overlay_path = None
    if canvas is not None:
        overlay_path = output_dir / f"{image_path.stem}_final_overlay.jpg"
        cv2.imwrite(str(overlay_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    json_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return overlay_path, json_path, result_payload


def main():
    args = parse_args()

    source = Path(args.source)
    det_model_path = Path(args.det_model)
    pose_model_path = Path(args.pose_model)
    output_dir = Path(args.output_dir)

    validate_model_path(det_model_path, "deteccion")
    validate_model_path(pose_model_path, "pose")

    images = collect_images(source)
    if not images:
        raise RuntimeError(f"No se encontraron imagenes para procesar en: {source}")

    det_model = YOLO(str(det_model_path))
    pose_model = YOLO(str(pose_model_path))

    print(f"Modelo deteccion: {det_model_path}")
    print(f"Modelo pose: {pose_model_path}")
    print(f"Imagenes a procesar: {len(images)}")

    for image_path in images:
        overlay_path, json_path, payload = process_image(
            det_model=det_model,
            pose_model=pose_model,
            image_path=image_path,
            output_dir=output_dir,
            det_conf=args.det_conf,
            pose_conf=args.pose_conf,
            padding=args.padding,
            json_only=args.json_only,
        )
        print(f"\nProcesada: {image_path.name}")
        if overlay_path is not None:
            print(f"  Overlay: {overlay_path}")
        print(f"  JSON: {json_path}")
        for side_name, side_info in payload["sides"].items():
            metrics = side_info["metrics"]
            if metrics.get("status") == "ok":
                print(
                    f"  {side_name}: LDFA={metrics['LDFA']:.2f} MPTA={metrics['MPTA']:.2f} "
                    f"HKA={metrics['HKA']:.2f} aHKA={metrics['aHKA']:.2f} JLO={metrics['JLO']:.2f} | {metrics['CPAK']}"
                )
            else:
                print(f"  {side_name}: sin calculo completo ({metrics})")


if __name__ == "__main__":
    main()
