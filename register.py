import os
import sys
import argparse
import gc
import numpy as np
import cv2
from insightface.app import FaceAnalysis

FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces")
FEATURES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.npz")
MAX_REGISTER_IMAGE_DIM = int(os.environ.get("MAX_REGISTER_IMAGE_DIM", "800"))
DET_SIZE = int(os.environ.get("REGISTER_DET_SIZE", "960"))
EXTRA_PERSON_BANDS = {
    "佐々木李子": ["sumimi"],
}


def person_bands(name, primary_band):
    bands = [primary_band]
    bands.extend(EXTRA_PERSON_BANDS.get(name, []))
    return sorted(dict.fromkeys(b for b in bands if b))


def encode_bands(bands):
    return ",".join(bands)


def load_insightface():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
    return app


def resize_for_registration(img):
    h, w = img.shape[:2]
    longest = max(w, h)
    if longest <= MAX_REGISTER_IMAGE_DIM:
        return img
    scale = MAX_REGISTER_IMAGE_DIM / longest
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def collect_person_vectors(app, person_dir):
    vecs = []
    for entry in sorted(os.scandir(person_dir), key=lambda e: e.name):
        if not entry.is_file():
            continue
        pp = entry.path
        if os.path.splitext(pp)[1].lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        try:
            img = cv2.imread(pp)
            if img is None:
                continue
            img = resize_for_registration(img)
            faces = app.get(img)
            if len(faces) == 0:
                print(f"  Warning: no face in {pp}, skipping")
                continue
            if len(faces) > 1:
                print(f"  Warning: {len(faces)} faces in {pp}, skipping")
                continue
            vecs.append(faces[0].normed_embedding.copy())
            del img, faces
        except Exception as exc:
            print(f"  Warning: failed to process {pp}: {exc}, skipping")
        finally:
            gc.collect()
    return vecs


def register(app, faces_dir):
    name_vecs = {}
    name_bands = {}
    for band_entry in sorted(os.scandir(faces_dir), key=lambda e: e.name):
        if not band_entry.is_dir():
            continue
        band = band_entry.name
        for person_entry in sorted(os.scandir(band_entry.path), key=lambda e: e.name):
            if not person_entry.is_dir():
                continue
            name = person_entry.name
            name_bands[name] = person_bands(name, band)
            name_vecs.setdefault(name, []).extend(
                collect_person_vectors(app, person_entry.path)
            )

    names = []
    bands = []
    feature_vectors = []
    for name in sorted(name_vecs):
        vecs = name_vecs[name]
        if not vecs:
            print(f"  Warning: no usable faces for {name}, skipping")
            continue
        mean_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        names.append(name)
        bands.append(encode_bands(name_bands.get(name, [])))
        feature_vectors.append(mean_vec)
        print(f"  Registered: {name} [{bands[-1]}] ({len(vecs)} photos)")
    return names, bands, np.array(feature_vectors) if feature_vectors else np.array([])


def register_one(app, faces_dir, band, name):
    person_dir = os.path.join(faces_dir, band, name)
    if not os.path.isdir(person_dir):
        for band_entry in sorted(os.scandir(faces_dir), key=lambda e: e.name):
            if not band_entry.is_dir():
                continue
            candidate = os.path.join(band_entry.path, name)
            if os.path.isdir(candidate):
                person_dir = candidate
                break
        else:
            print(f"Error: missing directory {person_dir}")
            sys.exit(1)
    source_band = os.path.basename(os.path.dirname(person_dir))
    vecs = collect_person_vectors(app, person_dir)
    if not vecs:
        print(f"Error: no usable faces for {band}/{name}")
        sys.exit(1)
    mean_vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    bands = person_bands(name, source_band)
    bands.append(band)
    encoded_bands = encode_bands(sorted(dict.fromkeys(bands)))
    print(f"  Registered one: {name} [{encoded_bands}] ({len(vecs)} photos)")
    return name, encoded_bands, mean_vec


def register_band(app, faces_dir, band):
    band_dir = os.path.join(faces_dir, band)
    if not os.path.isdir(band_dir):
        print(f"Error: missing directory {band_dir}")
        sys.exit(1)

    registered = []
    for person_entry in sorted(os.scandir(band_dir), key=lambda e: e.name):
        if not person_entry.is_dir():
            continue
        name = person_entry.name
        name, encoded_bands, vector = register_one(app, faces_dir, band, name)
        registered.append((name, encoded_bands, vector))

    if not registered:
        print(f"Error: no people found in {band_dir}")
        sys.exit(1)
    return registered


def upsert_feature(output, name, band, vector):
    if os.path.exists(output):
        data = np.load(output, allow_pickle=True)
        names = [str(n) for n in data["names"]]
        features = data["features"]
        if "bands" in data:
            bands = [str(b) for b in data["bands"]]
        else:
            bands = [""] * len(names)
    else:
        names = []
        bands = []
        features = np.empty((0, vector.shape[0]), dtype=vector.dtype)

    if name in names:
        idx = names.index(name)
        features[idx] = vector
        bands[idx] = band
        action = "Updated"
    else:
        names.append(name)
        bands.append(band)
        features = np.vstack([features, vector.reshape(1, -1)])
        action = "Inserted"

    np.savez(output, names=names, bands=bands, features=features)
    print(f"{action}: {name} [{band}] -> {output}")
    print(f"Saved {len(names)} feature vectors to {output}")


def main():
    parser = argparse.ArgumentParser(description="Register faces and save features (InsightFace)")
    parser.add_argument("-o", "--output", default=FEATURES_FILE)
    parser.add_argument("--band", help="Only register one band, or one band/person pair")
    parser.add_argument("--name", help="Only register one person with --band")
    args = parser.parse_args()

    print("Loading InsightFace buffalo_l...")
    app = load_insightface()

    if args.name and not args.band:
        print("Error: --name must be used with --band")
        sys.exit(1)

    if args.band and args.name:
        print(f"Registering one person from {FACES_DIR}/{args.band}/{args.name}...")
        name, band, vector = register_one(app, FACES_DIR, args.band, args.name)
        upsert_feature(args.output, name, band, vector)
        return

    if args.band:
        print(f"Registering band from {FACES_DIR}/{args.band}...")
        registered = register_band(app, FACES_DIR, args.band)
        for name, band, vector in registered:
            upsert_feature(args.output, name, band, vector)
        print(f"Registered band {args.band}: {[name for name, _, _ in registered]}")
        return

    print(f"Registering faces from {FACES_DIR}...")
    names, bands, feature_db = register(app, FACES_DIR)
    if len(names) == 0:
        print("Error: no faces registered")
        sys.exit(1)

    np.savez(args.output, names=names, bands=bands, features=feature_db)
    print(f"Saved {len(names)} feature vectors to {args.output}")
    print(f"Registered: {names}")


if __name__ == "__main__":
    main()
