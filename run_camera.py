import os
import sys
import argparse
import numpy as np
import cv2
import torch
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "AdaFace"))
import net as adaface_net
from face_alignment.mtcnn import MTCNN

FEATURES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.npz")
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt",
)

g_results = []
g_elapsed = 0


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


def recognize(mtcnn, adaface, names, feature_db, img_rgb):
    pil_img = Image.fromarray(img_rgb)
    boxes, aligned_faces = mtcnn.align_multi(pil_img)
    results = []
    for box, face_pil in zip(boxes, aligned_faces):
        vec = adaface_infer(adaface, np.array(face_pil))
        cos_results = feature_db @ vec / (
            np.linalg.norm(feature_db, axis=1) * np.linalg.norm(vec)
        )
        max_idx = int(np.argmax(cos_results))
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        results.append((names[max_idx], cos_results[max_idx], (x1, y1, x2, y2)))
    return results


def draw_results(image, results, elapsed):
    debug = image.copy()
    for name, score, (x1, y1, x2, y2) in results:
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            debug,
            f"{name} ({score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    cv2.putText(
        debug,
        f"{elapsed * 1000:.0f}ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    return debug


HD_720P = {"WIDTH": 1280, "HEIGHT": 720}


class Camera:
    def __init__(self, resolution=HD_720P):
        self.resolution = resolution
        self.camera_index = self.find_camera_index()
        self.cap = cv2.VideoCapture(self.camera_index)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution["WIDTH"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution["HEIGHT"])
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    @staticmethod
    def find_camera_index():
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                cap.release()
                return i
        raise ValueError("No camera found.")

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def get_frame(self):
        return self.frame

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while True:
                time.sleep(0.01)
                frame = camera.get_frame()
                if frame is not None:
                    annotated = draw_results(frame, g_results, g_elapsed)
                    ret, buf = cv2.imencode(".jpg", annotated)
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.end_headers()
                    self.wfile.write(buf.tobytes())
                    self.wfile.write(b"\r\n")
        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><head><title>Camera</title></head>"
                b"<body><h1>USB Camera Streaming</h1>"
                b'<img src="/video_feed"></body></html>'
            )

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera real-time face recognition")
    parser.add_argument("-f", "--features", default=FEATURES_FILE)
    parser.add_argument("--port", type=int, default=2233)
    args = parser.parse_args()

    print("Loading MTCNN...")
    mtcnn = load_mtcnn()

    print("Loading AdaFace...")
    adaface = load_adaface()

    print(f"Loading features from {args.features}...")
    data = np.load(args.features, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    feature_db = data["features"]

    camera = Camera(HD_720P)
    print("Camera initialized")

    server = HTTPServer(("0.0.0.0", args.port), StreamHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Stream at http://0.0.0.0:{args.port}")

    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t0 = time.time()
                g_results = recognize(mtcnn, adaface, names, feature_db, img_rgb)
                g_elapsed = time.time() - t0
            time.sleep(0.001)
    except KeyboardInterrupt:
        server.shutdown()
        camera.release()
        print("\nStopped.")
