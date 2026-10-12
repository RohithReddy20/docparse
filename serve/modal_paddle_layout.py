"""PP-DocLayoutV3 layout-detection server on Modal (PaddleOCR 3.x, Apache-2.0 weights).

Deploy:
    modal deploy serve/modal_paddle_layout.py
    # CPU-only variant (slower, ~free):  DOCPARSE_LAYOUT_GPU=cpu modal deploy serve/modal_paddle_layout.py

HTTP contract (what parse-bench's `paddle_layout` provider family expects):
    POST {url}/predict
        body: raw PNG/JPEG bytes (Content-Type: image/png)  -- HF-endpoint style
           or multipart form field `file`                   -- BYOC style
        -> {"predictions": [{"coordinate": [x1, y1, x2, y2],   # pixels of the input image
                             "label": "<PP-DocLayout 20-class name>",
                             "score": 0.97,
                             "raw_label": "<PP-DocLayoutV3 25-class name>",
                             "order": 0}, ...],              # list is in model reading order
            "image_width": W, "image_height": H, "model": "PP-DocLayoutV3"}
    GET {url}/health -> {"ok": true}

V3 adds 5 labels over the 20 parse-bench knows (display_formula, inline_formula,
vertical_text, header_image, footer_image, vision_footnote); they are folded into
the nearest 20-class label server-side (see V3_TO_PP20) and the original is kept
in `raw_label`.

Budget guards: one container max, 2-min idle scale-down, 1-hour hard timeout.
T4 is enough (~70 ms/page on A100; RT-DETR @ 800x800).
"""

import os
import re
import threading

import modal

GPU = os.environ.get("DOCPARSE_LAYOUT_GPU", "T4")
USE_GPU = GPU.lower() != "cpu"
MODEL_NAME = "PP-DocLayoutV3"
PADDLE_CACHE = "/root/.paddlex"

# PP-DocLayoutV3 (25) -> PP-DocLayout_plus-L (20) label names parse-bench maps.
V3_TO_PP20 = {
    "display_formula": "formula",
    "inline_formula": "formula",
    "vertical_text": "text",
    "header_image": "header",
    "footer_image": "footer",
    "vision_footnote": "footnote",
}

paddle_index = "https://www.paddlepaddle.org.cn/packages/stable/cu126/" if USE_GPU else \
    "https://www.paddlepaddle.org.cn/packages/stable/cpu/"
paddle_pkg = "paddlepaddle-gpu==3.3.0" if USE_GPU else "paddlepaddle==3.3.0"

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(paddle_pkg, index_url=paddle_index)
    .pip_install("paddleocr==3.7.0", "fastapi[standard]", "pillow", "python-multipart")
    .env({"PADDLE_PDX_MODEL_SOURCE": "HuggingFace"})
)

model_cache = modal.Volume.from_name("docparse-paddle-cache", create_if_missing=True)
app = modal.App("docparse-paddle-layout")


@app.cls(
    image=image,
    gpu=GPU if USE_GPU else None,
    cpu=4 if not USE_GPU else None,
    volumes={PADDLE_CACHE: model_cache},
    scaledown_window=120,
    timeout=3600,
    max_containers=1,
)
@modal.concurrent(max_inputs=8)
class LayoutServer:
    @modal.enter()
    def load(self) -> None:
        from paddleocr import LayoutDetection

        self.model = LayoutDetection(model_name=MODEL_NAME, device="gpu:0" if USE_GPU else "cpu")
        self.lock = threading.Lock()  # paddle predictors are not thread-safe
        model_cache.commit()

    def _predict(self, img_bytes: bytes) -> dict:
        import io

        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.asarray(img)[:, :, ::-1]  # paddle expects BGR
        with self.lock:
            results = list(self.model.predict(arr, batch_size=1, layout_nms=True))
        res = results[0].json if results else {}
        res = res.get("res", res)
        preds = []
        for i, box in enumerate(res.get("boxes", [])):
            raw = str(box.get("label", ""))
            coord = [float(v) for v in box.get("coordinate", [])]
            if len(coord) > 4:  # polygon -> axis-aligned box
                xs, ys = coord[0::2], coord[1::2]
                coord = [min(xs), min(ys), max(xs), max(ys)]
            preds.append(
                {
                    "coordinate": coord,
                    "label": V3_TO_PP20.get(raw, raw),
                    "score": float(box.get("score", 0.0)),
                    "raw_label": raw,
                    "order": i,
                }
            )
        return {"predictions": preds, "image_width": img.width, "image_height": img.height, "model": MODEL_NAME}

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, File, Request, UploadFile

        api = FastAPI()

        @api.get("/health")
        def health() -> dict:
            return {"ok": True, "model": MODEL_NAME, "gpu": GPU}

        @api.post("/predict")
        async def predict(request: Request, file: UploadFile | None = File(default=None)) -> dict:
            ctype = request.headers.get("content-type", "")
            if file is not None:
                data = await file.read()
            elif re.match(r"multipart/form-data", ctype):
                form = await request.form()
                up = form.get("file")
                data = await up.read()
            else:
                data = await request.body()
            return self._predict(data)

        # HF-endpoint style posts to the root URL
        api.add_api_route("/", predict, methods=["POST"])
        return api
