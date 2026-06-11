# Sistema-de-digitalizaci-n-de-documentos-hist-ricos-manuscritos

Pipeline de digitalización de actas civiles (texto impreso + manuscrito) usando Vision-Language Models. El sistema recibe una imagen de un acta, la preprocesa y extrae todos sus campos estructurados como JSON, exponiéndose como API REST desplegada en Kaggle con tunnel público via Cloudflare.

---

## Características principales

- **Preprocesamiento de imagen** con OpenCV: deskew (corrección de inclinación via Hough Lines), CLAHE (mejora de contraste adaptativo en espacio LAB) y sharpening por convolución.
- **Extracción estructurada de campos** en JSON con hasta 34 campos del acta (titular, padres, encargado, fechas, ubicación, etc.).
- **Dos versiones del modelo** probadas y comparadas:
  - `Qwen/Qwen2.5-VL-7B-Instruct` — inferencia greedy (`do_sample=False`)
  - `Qwen/Qwen3-VL-8B-Instruct` — inferencia con sampling (`temperature=0.7`, `top_p=0.8`, `top_k=20`)
- **API REST** con FastAPI (`/transcribir`, `/ping`) + CORS habilitado.
- **Almacenamiento automático** de resultados JSON en `/kaggle/working/resultados_actas/` con timestamp.
- **Tunnel público** via `cloudflared` para acceso externo desde Kaggle.

---

## Entorno de ejecución

El proyecto está diseñado para correr en **Kaggle Notebooks** con GPU T4 (2×T4 disponibles). No requiere GPU propia.

| Recurso | Valor |
|---|---|
| GPU | NVIDIA T4 (15 GB VRAM) |
| VRAM usada (Qwen2.5 fp16) | ~14 GB |
| Puerto API | `7860` |

---

## Instalación de dependencias

```bash
# Dependencias del servidor y modelo
pip install fastapi uvicorn python-multipart -q
pip install bitsandbytes -q
pip install "transformers>=4.49.0" -q      # para Qwen2.5-VL
# pip install git+https://github.com/huggingface/transformers -q  # para Qwen3-VL
pip install "qwen-vl-utils>=0.0.10" -q

# Cloudflared para el tunnel
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared
```

- El servidor ocupa el puerto `7860`. Si ya está en uso, se libera automáticamente con `fuser -k 7860/tcp` al inicio.
- Los JSON de resultados se guardan en `/kaggle/working/resultados_actas/` con nombre `acta_YYYYMMDD_HHMMSS.json`.
- La imagen preprocesada se devuelve en la respuesta como base64 PNG (redimensionada a máx. 900px de ancho) para visualización en el frontend.
- `torch.cuda.empty_cache()` se llama al final de cada inferencia para liberar VRAM entre requests.
