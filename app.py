import os
import base64
import io
from flask import Flask, request, jsonify, send_from_directory
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")

VISION_KEY = os.environ.get("VISION_KEY", "")
VISION_ENDPOINT = os.environ.get("VISION_ENDPOINT", "").rstrip("/")


FACE_KEY = os.environ.get("FACE_KEY", "")
FACE_ENDPOINT = os.environ.get("FACE_ENDPOINT", "").rstrip("/")

# Lazily-loaded OpenCV face cascade (used as a free, local fallback for face
# detection when no Azure Face API key is configured, or when that call fails).
_FACE_CASCADE = None


def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        import cv2
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def _get_image_bytes(image_url, image_base64):
    """Return raw image bytes regardless of which input was supplied."""
    if image_base64:
        b64 = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
        return base64.b64decode(b64)
    resp = requests.get(image_url, timeout=20)
    resp.raise_for_status()
    return resp.content


def _detect_faces_local(image_url, image_base64):
    """Local, offline face detection using OpenCV Haar cascades.

    Works without any Azure key/quota, so it always provides at least
    bounding-box face detection as a fallback or independent confirmation.
    Returns a list of {boundingBox, confidence} dicts.
    """
    try:
        import cv2
        import numpy as np

        img_bytes = _get_image_bytes(image_url, image_base64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        cascade = _get_face_cascade()
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        results = []
        for (x, y, w, h) in faces:
            results.append({
                "boundingBox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "confidence": 0.85,  # Haar cascades don't return a real score
                "source": "opencv-local",
            })
        return results
    except Exception:
        return []


PERSON_GROUP_ID = os.environ.get("FACE_PERSON_GROUP_ID", "family-members")
_person_group_ready = False


def _face_headers():
    return {"Ocp-Apim-Subscription-Key": FACE_KEY, "Content-Type": "application/json"}


def _ensure_person_group():
    """Create the family PersonGroup once if it doesn't already exist."""
    global _person_group_ready
    if _person_group_ready or not (FACE_KEY and FACE_ENDPOINT):
        return
    url = f"{FACE_ENDPOINT}/face/v1.0/persongroups/{PERSON_GROUP_ID}"
    try:
        resp = requests.get(url, headers=_face_headers(), timeout=10)
        if resp.status_code == 404:
            requests.put(
                url, headers=_face_headers(), timeout=10,
                json={"name": "Family Members", "recognitionModel": "recognition_04"},
            )
        _person_group_ready = True
    except requests.RequestException:
        pass


def list_people():
    """Return list of enrolled people: [{personId, name, numFaces}]."""
    if not (FACE_KEY and FACE_ENDPOINT):
        return []
    _ensure_person_group()
    url = f"{FACE_ENDPOINT}/face/v1.0/persongroups/{PERSON_GROUP_ID}/persons"
    try:
        resp = requests.get(url, headers=_face_headers(), timeout=10)
        if resp.status_code != 200:
            return []
        return [
            {"personId": p["personId"], "name": p["name"],
             "numFaces": len(p.get("persistedFaceIds", []))}
            for p in resp.json()
        ]
    except requests.RequestException:
        return []


def enroll_person(name, image_url, image_base64):
    """Create (or reuse) a person by name and add a face photo to them."""
    if not (FACE_KEY and FACE_ENDPOINT):
        return {"error": "Face API not configured"}, 500
    _ensure_person_group()

    # Reuse an existing person with the same name instead of creating dupes
    existing = next((p for p in list_people() if p["name"].lower() == name.lower()), None)
    if existing:
        person_id = existing["personId"]
    else:
        create_url = f"{FACE_ENDPOINT}/face/v1.0/persongroups/{PERSON_GROUP_ID}/persons"
        resp = requests.post(create_url, headers=_face_headers(), json={"name": name}, timeout=10)
        if resp.status_code not in (200, 201):
            return {"error": f"Could not create person: {resp.text}"}, resp.status_code
        person_id = resp.json()["personId"]

    add_face_url = (
        f"{FACE_ENDPOINT}/face/v1.0/persongroups/{PERSON_GROUP_ID}"
        f"/persons/{person_id}/persistedfaces"
    )
    headers = dict(_face_headers())
    if image_url:
        resp = requests.post(add_face_url, headers=headers, json={"url": image_url}, timeout=20)
    else:
        b64 = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
        headers["Content-Type"] = "application/octet-stream"
        resp = requests.post(add_face_url, headers=headers, data=base64.b64decode(b64), timeout=20)

    if resp.status_code not in (200, 201):
        return {"error": f"Could not add face: {resp.text}"}, resp.status_code

    return {"personId": person_id, "name": name, "status": "face added"}, 200


def delete_person(person_id):
    if not (FACE_KEY and FACE_ENDPOINT):
        return False
    url = f"{FACE_ENDPOINT}/face/v1.0/persongroups/{PERSON_GROUP_ID}/persons/{person_id}"
    try:
        resp = requests.delete(url, headers=_face_headers(), timeout=10)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def train_person_group():
    """Kick off (re)training; the model must be trained after enrolling faces
    before identification will reflect the new data."""
    if not (FACE_KEY and FACE_ENDPOINT):
        return {"error": "Face API not configured"}
    _ensure_person_group()
    url = f"{FACE_ENDPOINT}/face/v1.0/persongroups/{PERSON_GROUP_ID}/train"
    try:
        resp = requests.post(url, headers=_face_headers(), timeout=10)
        return {"status": "training started" if resp.status_code == 202 else "error",
                "code": resp.status_code}
    except requests.RequestException as exc:
        return {"error": str(exc)}


def training_status():
    if not (FACE_KEY and FACE_ENDPOINT):
        return {"error": "Face API not configured"}
    url = f"{FACE_ENDPOINT}/face/v1.0/persongroups/{PERSON_GROUP_ID}/training"
    try:
        resp = requests.get(url, headers=_face_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "notstarted"}
    except requests.RequestException as exc:
        return {"error": str(exc)}


def identify_faces(face_ids):
    """Match detected faceIds against the trained PersonGroup.
    Returns {faceId: {"name": str, "confidence": float} or None}."""
    if not face_ids or not (FACE_KEY and FACE_ENDPOINT):
        return {}
    url = f"{FACE_ENDPOINT}/face/v1.0/identify"
    body = {
        "personGroupId": PERSON_GROUP_ID,
        "faceIds": face_ids,
        "maxNumOfCandidatesReturned": 1,
        "confidenceThreshold": 0.5,
    }
    try:
        resp = requests.post(url, headers=_face_headers(), json=body, timeout=20)
        if resp.status_code != 200:
            return {}
        people = {p["personId"]: p["name"] for p in list_people()}
        matches = {}
        for entry in resp.json():
            candidates = entry.get("candidates", [])
            if candidates:
                pid = candidates[0]["personId"]
                matches[entry["faceId"]] = {
                    "name": people.get(pid, "Unknown"),
                    "confidence": candidates[0]["confidence"],
                }
            else:
                matches[entry["faceId"]] = None
        return matches
    except requests.RequestException:
        return {}


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


def _call_vision(image_url, image_base64, features="Tags,Read,Objects,People"):
    """Call Azure AI Vision Image Analysis 4.0 and return (body, status_code)."""
    api_url = (
        f"{VISION_ENDPOINT}/computervision/imageanalysis:analyze"
        f"?api-version=2023-10-01&features={features}"
    )
    headers = {"Ocp-Apim-Subscription-Key": VISION_KEY}

    if image_url:
        headers["Content-Type"] = "application/json"
        resp = requests.post(api_url, headers=headers, json={"url": image_url}, timeout=20)
    else:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        headers["Content-Type"] = "application/octet-stream"
        resp = requests.post(api_url, headers=headers, data=image_bytes, timeout=20)

    try:
        body = resp.json()
    except ValueError:
        body = {"error": resp.text or "Unexpected response from Azure AI Vision"}
    return body, resp.status_code


def _call_face_api(image_url, image_base64):
    """Call Azure Face API for rich face attributes. Returns list of faces or None."""
    if not FACE_KEY or not FACE_ENDPOINT:
        return None

    face_url = (
        f"{FACE_ENDPOINT}/face/v1.0/detect"
        "?returnFaceAttributes=age,gender,emotion,glasses,headPose,smile,facialHair,makeup"
        "&returnFaceId=true"
        "&returnFaceLandmarks=false"
        "&recognitionModel=recognition_04"
        "&detectionModel=detection_01"
    )
    headers = {"Ocp-Apim-Subscription-Key": FACE_KEY}

    try:
        if image_url:
            headers["Content-Type"] = "application/json"
            resp = requests.post(face_url, headers=headers, json={"url": image_url}, timeout=20)
        else:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_base64)
            headers["Content-Type"] = "application/octet-stream"
            resp = requests.post(face_url, headers=headers, data=image_bytes, timeout=20)

        if resp.status_code == 200:
            faces = resp.json()
            # Try to recognize each detected face against enrolled family members
            face_ids = [f["faceId"] for f in faces if f.get("faceId")]
            matches = identify_faces(face_ids)
            for f in faces:
                match = matches.get(f.get("faceId"))
                f["recognized"] = match  # {"name", "confidence"} or None
            return faces
    except requests.RequestException:
        pass
    return None


def _call_vision_v32_brands_landmarks(image_url, image_base64):
    """Call the Azure Computer Vision v3.2 'analyze' endpoint to get the
    real Brands feature and the Landmarks domain-specific model.

    These are dedicated, trained recognition models (not keyword guessing)
    that Azure's newer v4.0 Image Analysis API doesn't expose, so we hit the
    v3.2 endpoint on the same Cognitive Services resource for this data.
    Returns (brands, landmarks) as plain lists, or ([], []) on any failure.
    """
    if not VISION_KEY or not VISION_ENDPOINT:
        return [], []

    api_url = (
        f"{VISION_ENDPOINT}/vision/v3.2/analyze"
        "?visualFeatures=Brands&details=Landmarks"
    )
    headers = {"Ocp-Apim-Subscription-Key": VISION_KEY}

    try:
        if image_url:
            headers["Content-Type"] = "application/json"
            resp = requests.post(api_url, headers=headers, json={"url": image_url}, timeout=20)
        else:
            b64 = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
            image_bytes = base64.b64decode(b64)
            headers["Content-Type"] = "application/octet-stream"
            resp = requests.post(api_url, headers=headers, data=image_bytes, timeout=20)

        if resp.status_code != 200:
            return [], []

        result = resp.json()

        brands = [
            {
                "name": b.get("name", ""),
                "confidence": b.get("confidence", 0),
                "boundingBox": b.get("rectangle"),
                "source": "azure-brands-model",
            }
            for b in result.get("brands", [])
        ]

        landmarks = []
        for cat in result.get("categories", []):
            for lm in cat.get("detail", {}).get("landmarks", []):
                landmarks.append({
                    "name": lm.get("name", ""),
                    "confidence": lm.get("confidence", 0),
                    "source": "azure-landmark-model",
                })

        return brands, landmarks
    except requests.RequestException:
        return [], []


@app.route("/analyze", methods=["POST"])
def analyze():
    if not VISION_KEY or not VISION_ENDPOINT:
        return jsonify({
            "error": "Azure Vision not configured. Set VISION_KEY and "
                     "VISION_ENDPOINT in App Service -> Configuration -> "
                     "Application settings, then restart the app."
        }), 500

    data = request.get_json(silent=True) or {}
    image_url = data.get("url")
    image_base64 = data.get("image_base64")

    if not image_url and not image_base64:
        return jsonify({"error": "No image URL or image data provided"}), 400

    try:
        # Vision API: Tags, Read (OCR), Objects (includes brands), People (face boxes)
        body, status = _call_vision(image_url, image_base64,
                                    features="Tags,Read,Objects,People")
    except requests.RequestException as exc:
        return jsonify({"error": f"Request to Azure AI Vision failed: {exc}"}), 502

    # Enrich with Face API attributes if available; always also run the free
    # local OpenCV face detector so faces still show up when no Face API key
    # is configured, when the key has hit its quota, or as a sanity check.
    face_details = _call_face_api(image_url, image_base64)
    if face_details is not None:
        body["faceApiResult"] = face_details

    local_faces = _detect_faces_local(image_url, image_base64)
    body["localFacesDetected"] = local_faces

    # Real brand + landmark recognition models (Azure v3.2 domain models)
    real_brands, real_landmarks = _call_vision_v32_brands_landmarks(image_url, image_base64)

    # Extract landmark and brand tags from Objects + tag hints
    landmarks = list(real_landmarks)
    brands = list(real_brands)
    objects = body.get("objectsResult", {}).get("values", [])
    for obj in objects:
        tags_list = obj.get("tags", [])
        for t in tags_list:
            name = t.get("name", "")
            conf = t.get("confidence", 0)
            # Azure surfaces landmarks and brands as objects with specific tag names
            # We surface them all; the frontend filters by label keyword
            entry = {"name": name, "confidence": conf,
                     "boundingBox": obj.get("boundingBox"), "source": "heuristic"}
            if any(kw in name.lower() for kw in ["tower", "palace", "monument",
                                                   "bridge", "landmark", "cathedral",
                                                   "castle", "statue", "temple"]):
                if not any(l["name"].lower() == name.lower() for l in landmarks):
                    landmarks.append(entry)
            elif conf > 0.5 and name[0].isupper():
                if not any(b["name"].lower() == name.lower() for b in brands):
                    brands.append(entry)

    # Also check tags for well-known landmark/brand hints
    tag_values = body.get("tagsResult", {}).get("values", [])
    for tag in tag_values:
        name = tag.get("name", "")
        conf = tag.get("confidence", 0)
        if conf > 0.7 and any(kw in name.lower() for kw in
                               ["tower", "palace", "monument", "bridge",
                                "landmark", "cathedral", "castle", "statue", "temple"]):
            if not any(l["name"].lower() == name.lower() for l in landmarks):
                landmarks.append({"name": name, "confidence": conf, "source": "heuristic"})

    body["landmarksDetected"] = landmarks
    body["brandsDetected"] = brands

    return jsonify(body), status


@app.route("/people", methods=["GET"])
def get_people():
    return jsonify({"people": list_people(), "personGroupId": PERSON_GROUP_ID})


@app.route("/people", methods=["POST"])
def add_person():
    if not (FACE_KEY and FACE_ENDPOINT):
        return jsonify({"error": "Face API not configured. Set FACE_KEY and "
                                  "FACE_ENDPOINT to enable face recognition."}), 500

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    image_url = data.get("url")
    image_base64 = data.get("image_base64")

    if not name:
        return jsonify({"error": "A name is required"}), 400
    if not image_url and not image_base64:
        return jsonify({"error": "No image URL or image data provided"}), 400

    result, status = enroll_person(name, image_url, image_base64)
    return jsonify(result), status


@app.route("/people/<person_id>", methods=["DELETE"])
def remove_person(person_id):
    ok = delete_person(person_id)
    return jsonify({"deleted": ok}), (200 if ok else 500)


@app.route("/people/train", methods=["POST"])
def train_people():
    return jsonify(train_person_group())


@app.route("/people/train/status", methods=["GET"])
def train_people_status():
    return jsonify(training_status())


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "configured": bool(VISION_KEY and VISION_ENDPOINT),
        "faceApiConfigured": bool(FACE_KEY and FACE_ENDPOINT),
        "faceRecognitionAvailable": bool(FACE_KEY and FACE_ENDPOINT),
        "localFaceDetectionAvailable": True,
    })


if __name__ == "__main__":
    app.run(debug=True)
