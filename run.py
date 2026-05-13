import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image

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


def recognize(mtcnn, adaface, names, feature_db, image):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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


def draw_results(image, results):
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
    return debug


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single image face recognition")
    parser.add_argument("-i", "--image", required=True, help="Input image path")
    parser.add_argument("-f", "--features", default=FEATURES_FILE)
    parser.add_argument(
        "-o", "--output", default=None, help="Output image path (default: overwrite)"
    )
    args = parser.parse_args()

    print("Loading MTCNN...")
    mtcnn = load_mtcnn()

    print("Loading AdaFace...")
    adaface = load_adaface()

    print(f"Loading features from {args.features}...")
    data = np.load(args.features, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    feature_db = data["features"]

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: cannot read {args.image}")
        sys.exit(1)

    results = recognize(mtcnn, adaface, names, feature_db, img)
    for name, score, bbox in results:
        print(f"  {name} ({score:.3f})")

    output = draw_results(img, results)
    out_path = args.output or args.image
    cv2.imwrite(out_path, output)
    print(f"Saved to {out_path}")
