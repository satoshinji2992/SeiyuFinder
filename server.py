import os
import sys
import socket
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import time
import threading

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
HISTORY_FILE = os.path.join(UPLOADS_DIR, "history.json")
os.makedirs(UPLOADS_DIR, exist_ok=True)

history_lock = threading.Lock()


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "AdaFace"))
import net as adaface_net
from face_alignment.mtcnn import MTCNN

FEATURES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.npz")
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt",
)


def load_adaface():
    model = adaface_net.build_model("ir_50")
    statedict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    model.load_state_dict(
        {k[6:]: v for k, v in statedict.items() if k.startswith("model.")}
    )
    model.eval()
    return model


def load_mtcnn():
    m = MTCNN(device="cpu", crop_size=(112, 112))
    m.min_face_size = 12
    m.thresholds = [0.4, 0.5, 0.7]
    return m


def adaface_infer(model, face_aligned):
    bgr = ((face_aligned[:, :, ::-1] / 255.0) - 0.5) / 0.5
    tensor = torch.tensor(np.array([bgr.transpose(2, 0, 1)])).float()
    with torch.no_grad():
        feature, _ = model(tensor)
    return feature[0].numpy()


def recognize(mtcnn, adaface, names, feature_db, image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_raw is None:
        return []
    pil_img = Image.fromarray(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    _, aligned_faces = mtcnn.align_multi(pil_img)

    results = []
    for face_pil in aligned_faces:
        vec = adaface_infer(adaface, np.array(face_pil))
        cos_results = feature_db @ vec / (
            np.linalg.norm(feature_db, axis=1) * np.linalg.norm(vec)
        )
        max_idx = int(np.argmax(cos_results))
        results.append(names[max_idx])
    return results


class FaceHandler(BaseHTTPRequestHandler):
    mtcnn = None
    adaface = None
    names = None
    feature_db = None

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "index.html"
            )
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())
        elif self.path.startswith("/avatar/"):
            name = urllib.parse.unquote(self.path[len("/avatar/"):])
            faces_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "faces", name
            )
            if os.path.isdir(faces_dir):
                for ext in [".jpg", ".jpeg", ".png"]:
                    photo = os.path.join(faces_dir, "1" + ext)
                    if os.path.exists(photo):
                        self.send_response(200)
                        ct = "image/jpeg" if ext != ".png" else "image/png"
                        self.send_header("Content-Type", ct)
                        self.end_headers()
                        with open(photo, "rb") as f:
                            self.wfile.write(f.read())
                        return
            self.send_response(404)
            self.end_headers()
        elif self.path == "/people":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                json.dumps({"names": self.names}, ensure_ascii=False).encode("utf-8")
            )
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            result = recognize(
                self.mtcnn, self.adaface, self.names, self.feature_db, body
            )
            ts = int(time.time() * 1000)
            photo_name = f"{ts}.jpg"
            photo_path = os.path.join(UPLOADS_DIR, photo_name)
            nparr = np.frombuffer(body, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imwrite(photo_path, img, [cv2.IMWRITE_JPEG_QUALITY, 10])
                with history_lock:
                    history = load_history()
                    history.append({
                        "photo": f"uploads/{photo_name}",
                        "faces": result,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    save_history(history)
            payload = json.dumps({"faces": result}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def log_message(self, format, *args):
        print(f"[server] {args[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3724)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("-f", "--features", default=FEATURES_FILE)
    args = parser.parse_args()

    print("Loading MTCNN...")
    mtcnn = load_mtcnn()

    print("Loading AdaFace...")
    adaface = load_adaface()

    print(f"Loading features from {args.features}...")
    data = np.load(args.features, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    feature_db = data["features"]
    print(f"Loaded {len(names)} people, feature dim: {feature_db.shape[1]}")

    FaceHandler.mtcnn = mtcnn
    FaceHandler.adaface = adaface
    FaceHandler.names = names
    FaceHandler.feature_db = feature_db

    server = HTTPServer((args.host, args.port), FaceHandler, False)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.server_bind()
    server.server_activate()
    print(f"Serving on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
