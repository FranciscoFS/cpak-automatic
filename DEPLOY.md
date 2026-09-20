# 🚀 Guía de despliegue — Streamlit Community Cloud (gratis)

Esta carpeta es un repositorio **autocontenido** listo para desplegar.

```text
deploy_app/
├── README.md              # documentación
├── requirements.txt       # deps Python (torch CPU-only)  ← lo detecta la nube
├── packages.txt           # paquetes apt del sistema (libGL para OpenCV)
├── Dockerfile             # alternativa: HF Spaces (PRO) / Cloud Run / Render
├── .dockerignore
├── .gitignore
├── LICENSE                # AGPL-3.0
├── NOTICE.md              # licencias de terceros
├── DEPLOY.md              # este archivo
├── scripts/
│   ├── 30_streamlit_app.py    # entrypoint de la app
│   └── 13_final_inference.py  # pipeline
└── training_runs/
    ├── telerx_yolo26s_768_b12-4/weights/best.pt
    └── telerx_pose_v2_ft/weights/best.pt
```

> ⚠️ La estructura de carpetas **debe** mantenerse: el código resuelve rutas con
> `Path(__file__).resolve().parents[1]`, por lo que los modelos deben quedar en
> `training_runs/...` y el pipeline en `scripts/`.

---

## 0. Prueba local antes de subir (recomendado)

```powershell
cd deploy_app
conda create -n cpak_deploy python=3.10 -y
conda activate cpak_deploy
pip install -r requirements.txt
streamlit run scripts/30_streamlit_app.py
```

Se abre en <http://localhost:8501>. Confirma que carga una radiografía y que
aparecen las métricas antes de desplegar.

---

## 1. Subir el proyecto a GitHub

> ⚠️ **Este kit debe ser su propio repositorio.** Streamlit Community Cloud busca
> `requirements.txt` en la raíz del repo o junto al entrypoint — si lo metes
> dentro del repo de investigación, tomaría el `requirements.txt` equivocado.

### 1a. Crea el repositorio vacío

En <https://github.com/new>:

- **Repository name**: `cpak-automatic`
- **Visibility**: Public
- **NO** marques "Add a README file" (ya tienes uno)

### 1b. Inicializa y sube

Desde esta carpeta (`deploy_app/`):

```powershell
git init
git add -A
git commit -m "CPAK automatic coronal alignment app"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/cpak-automatic.git
git push -u origin main
```

Si te pide credenciales, usa tu usuario de GitHub y un **token** con permisos
`repo` (<https://github.com/settings/tokens>) como contraseña.

> Los pesos (`.pt`, ~40 MB en total) van **directamente** en el repo: están por
> debajo del límite de 100 MB de GitHub, así que **no** se necesita Git LFS.

---

## 2. Desplegar en Streamlit Community Cloud

1. Entra en 👉 <https://share.streamlit.io> y haz **Sign in with GitHub**.
2. Autoriza la app (para repos públicos basta el OAuth por defecto).
3. Pulsa **Create app** → **Deploy a public app from GitHub**.
4. Rellena:
   - **Repository**: `<tu-usuario>/cpak-automatic`
   - **Branch**: `main`
   - **Main file path**: `scripts/30_streamlit_app.py`
5. (Opcional) **Advanced settings**: elige la versión de Python si lo necesitas.
6. Pulsa **Deploy!**

La URL quedará como `https://<nombre-app>.streamlit.app`.

Como el entrypoint está en `scripts/`, la nube buscará dependencias primero en
`scripts/` y después en la raíz → encontrará `requirements.txt` y `packages.txt`
de la raíz. ✅

---

## 3. Verificar

Espera a que el log termine el build (instalar torch CPU tarda unos minutos la
primera vez). Luego prueba:

1. Subir una radiografía anonimizada.
2. Pegar una captura con `Ctrl+V`.
3. Presionar **▶ Ejecutar inferencia** y revisar métricas + overlay.

Cada `git push` a `main` redespliega automáticamente.

---

## 4. Solución de problemas

| Síntoma | Causa probable | Solución |
| :--- | :--- | :--- |
| El build descarga varios GB | Falta la línea `--extra-index-url` | Debe ser la **primera** línea de `requirements.txt` |
| `ImportError: libGL.so.1` | Falta `packages.txt` | Verifica que esté en la **raíz** del repo |
| `ImportError: libgthread-2.0.so.0` | Falta libglib | `libglib2.0-0` en `packages.txt` |
| `ModuleNotFoundError: ultralytics` | No encontró el `requirements.txt` | Debe estar en la raíz del repo (o junto al entrypoint) |
| La app se reinicia / "Oh no" al procesar | **Límite de RAM** de la nube | Ver nota de memoria más abajo |
| `ERROR: No matching distribution found for torch==2.10.0+cpu` | El índice CPU no se está usando | Confirma el `--extra-index-url` |
| Error de `streamlit-drawable-canvas` | Cambio de API en Streamlit | La app lo maneja y desactiva solo el editor de puntos |

### ⚠️ Nota sobre memoria (importante)

Medimos el consumo real de la app: **pico ≈900 MB** con una radiografía de 20 MB
(modelos cargados + inferencia). El plan gratuito de Streamlit Community Cloud
tiene memoria acotada, así que:

- Con radiografías normales (2–5 MB) debería ir bien.
- Si la app se cae por memoria, hay dos mitigaciones:
  1. Convertir los modelos a **ONNX** y usar `onnxruntime` en lugar de
     `torch` + `ultralytics`: baja el consumo a ~150–300 MB (requiere reexportar
     los `.pt` y ajustar `13_final_inference.py`).
  2. Usar un host con más RAM (ver alternativas).

---

## 5. Alternativas de hosting

| Host | Estado | Notas |
| :--- | :--- | :--- |
| **Streamlit Community Cloud** | ✅ **Gratis** | La opción recomendada. Duerme por inactividad. |
| **Hugging Face Spaces** | ⚠️ Requiere **PRO** | El SDK *Streamlit* ya **no existe** en la API (solo `gradio`, `docker`, `static`), y `gradio`/`docker` en CPU básica exigen suscripción PRO. Con PRO, usa el `Dockerfile` incluido: al añadir el frontmatter `sdk: docker` + `app_port: 7860` al `README.md`, sube la carpeta como Space. |
| Google Cloud Run | ✅ Free tier | Requiere cuenta de facturación. Usa el `Dockerfile` incluido (el puerto es el 8501/7860). |
| Cloudflare Tunnel / ngrok | ✅ Gratis, **privado** | Nada sale de tu equipo; solo online mientras tu PC esté encendido. Ideal para demos a un comité sin exponer datos. |

### Compartir de forma privada con un túnel

```powershell
# en una terminal: la app local
conda run -n physis_seg streamlit run scripts/30_streamlit_app.py --server.port 8501

# en otra: exponer una URL temporal
cloudflared tunnel --url http://localhost:8501
```

---

## 6. Checklist antes de publicar

- [ ] **Sin datos de pacientes** en el repositorio (ninguna radiografía real)
- [ ] `LICENSE` completo (descargar el texto íntegro de AGPL-3.0; ver `LICENSE`)
- [x] Aviso de privacidad visible en la app ✅ (al inicio de la barra lateral)
- [ ] Enlace al repositorio de código fuente (obligatorio bajo AGPL §13)
- [ ] Revisar `README.md`: autor(es), afiliación, estado de validación

### Aviso de privacidad dentro de la app (ya implementado)

Se muestra al inicio de la barra lateral de `scripts/30_streamlit_app.py`:

```python
with st.sidebar:
    st.warning(
        "**Demo de investigación.** No subas datos identificables de pacientes "
        "(nombres, RUT, fechas de nacimiento). No es un dispositivo médico "
        "certificado: los resultados deben ser revisados por un cirujano."
    )
```

Si quieres cambiarlo, edita ese bloque y vuelve a copiar el archivo al repo.
