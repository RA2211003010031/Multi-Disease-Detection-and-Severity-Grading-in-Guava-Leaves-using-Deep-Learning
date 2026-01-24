import torch
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- LOAD MODEL ----------------
model = YOLO("best.pt")

disease_info = {
    "Anthracnose": {
        "description": "A fungal disease caused by Colletotrichum species that creates dark, sunken lesions on guava leaves and fruits.",
        "cause": "High humidity, rainfall, and fungal spores surviving on infected plant debris.",
        "impact": "Reduces photosynthesis, causes leaf drop, and leads to fruit rot, lowering yield and market value.",
        "treatment": "Spray Carbendazim (1g/L) or Copper Oxychloride every 10–14 days during outbreaks.",
        "organic": "Apply neem oil or Trichoderma-based biofungicide.",
        "prevention": "Prune infected branches, avoid overhead irrigation, and ensure good air circulation.",
        "future_safety": "Plant resistant varieties such as Allahabad Safeda and apply pre-monsoon fungicide schedules."
    },
    "Nutrient_Deficiency": {
        "description": "A physiological disorder where the plant lacks essential nutrients such as nitrogen, iron, or zinc.",
        "cause": "Poor soil fertility, imbalanced fertilizer use, or improper pH.",
        "impact": "Leads to yellowing (chlorosis), weak growth, and reduced fruit size.",
        "treatment": "Apply balanced NPK fertilizer and foliar sprays of iron and zinc.",
        "organic": "Use vermicompost, seaweed extract, or compost tea.",
        "prevention": "Conduct soil testing and follow nutrient management plans.",
        "future_safety": "Install drip fertigation systems for precise nutrient delivery."
    },
    "Wilt": {
        "description": "A soil-borne fungal disease that blocks water movement inside the plant.",
        "cause": "Pathogens like Fusarium in poorly drained soil.",
        "impact": "Sudden wilting, yellowing, and plant death.",
        "treatment": "Soil drenching with carbendazim or fungicides.",
        "organic": "Apply neem cake and Trichoderma in soil.",
        "prevention": "Improve drainage and avoid replanting in infected soil.",
        "future_safety": "Use resistant rootstocks."
    },
    "Insect_Attack": {
        "description": "Damage caused by insects such as fruit flies, aphids, or mealybugs.",
        "cause": "Warm weather and poor orchard hygiene.",
        "impact": "Leaf curling, fruit drop, and secondary infections.",
        "treatment": "Use neem oil or Dimethoate as needed.",
        "organic": "Introduce ladybirds and lacewings.",
        "prevention": "Install pheromone traps and remove infected fruits.",
        "future_safety": "Use integrated pest management (IPM)."
    },
    "Healthy": {
        "description": "The leaf shows normal color, texture, and structure.",
        "cause": "Proper nutrition and environment.",
        "impact": "Good growth and high yield potential.",
        "treatment": "No treatment required.",
        "organic": "Maintain compost and mulch.",
        "prevention": "Continue good farming practices.",
        "future_safety": "Keep digital growth and health records."
    }
}

model.eval()

CONF_THRESH = 0.45

# ---------------- SEVERITY UTILS ----------------
def classify(sev):
    if sev < 5:
        return "Healthy"
    elif sev < 20:
        return "Mild"
    elif sev < 50:
        return "Moderate"
    else:
        return "Severe"
    
def severity_color(sev):
    if sev < 5:
        return (0, 255, 0)        # Green
    elif sev < 20:
        return (0, 255, 255)      # Yellow
    elif sev < 50:
        return (0, 165, 255)      # Orange
    else:
        return (0, 0, 255)        # Red

# ---------------- YOLO + SEVERITY ----------------
def run_yolo(img):
    raw = img.copy()
    output = img.copy()
    h, w = img.shape[:2]

    boxes = []
    detected_diseases = set()
    disease_mask = np.zeros((h, w), dtype=np.uint8)

    # ---------- YOLO OBB DETECTION ----------
    results = model(img)[0]

    if results.obb is not None:
        polys = results.obb.xyxyxyxy.cpu().numpy()
        classes = results.obb.cls.cpu().numpy()
        confs = results.obb.conf.cpu().numpy()

        for poly, cls, conf in zip(polys, classes, confs):
            if conf < CONF_THRESH:
                continue

            pts = poly.reshape(4, 2).astype(np.int32)
            label = results.names[int(cls)]

            detected_diseases.add(label)

            area = cv2.contourArea(pts)
            boxes.append({
                "label": label,
                "conf": float(conf),
                "points": pts,
                "area": area
            })

            cv2.fillPoly(disease_mask, [pts], 255)
            cv2.polylines(output, [pts], True, (0, 255, 0), 2)

    # ---------- PER-BOX LABELS ----------
    for b in boxes:
        cx = int(b["points"][:, 0].mean())
        cy = int(b["points"][:, 1].mean())

        text = f"{b['label']} {b['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

        cv2.rectangle(
            output,
            (cx - tw // 2 - 4, cy - th - 6),
            (cx + tw // 2 + 4, cy),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            output,
            text,
            (cx - tw // 2, cy - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ---------- LEAF SEGMENTATION ----------
    hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    leaf_mask = cv2.inRange(hsv, (25, 30, 30), (90, 255, 255))

    cnts, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        leaf_mask[:] = 0
        cv2.drawContours(leaf_mask, [c], -1, 255, -1)

    leaf_area = cv2.countNonZero(leaf_mask)
    disease_area = cv2.countNonZero(cv2.bitwise_and(disease_mask, leaf_mask))

    # ---------- GLOBAL SEVERITY ----------
    severity = (disease_area / leaf_area) * 100 if leaf_area > 0 else 0.0
    level = classify(severity)
    color = severity_color(severity)

    # # ---------- FOOTER ----------
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.8
    # thickness = 2
    # footer_height = 35

    # text_left = "Damage: "
    # text_right = f"{severity:.2f}% ({level})"

    # (lw, lh), _ = cv2.getTextSize(text_left, font, font_scale, thickness)
    # (rw, _), _ = cv2.getTextSize(text_right, font, font_scale, thickness)

    # x = (w - (lw + rw)) // 2
    # y = h - (footer_height // 2) + (lh // 2)

    # cv2.rectangle(output, (0, h - footer_height), (w, h), (0, 0, 0), -1)
    # cv2.putText(output, text_left, (x, y), font, font_scale, (255, 255, 255), thickness)
    # cv2.putText(output, text_right, (x + lw, y), font, font_scale, color, thickness)

    if not detected_diseases:
        detected_diseases = {"Healthy"}

    return output, severity, level, list(detected_diseases), results, boxes


# ---------------- PER-DISEASE GRAD-CAM (OPTION-A) ----------------
def run_gradcam(img, boxes, target_layer=10):
    device = next(model.model.parameters()).device
    was_training = model.model.training
    model.model.train()

    img_resized = cv2.resize(img, (640, 640))
    inp = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    inp.requires_grad_(True)

    activations, gradients = [], []

    def fwd_hook(m, i, o):
        activations.append(o)

    def bwd_hook(m, gi, go):
        gradients.append(go[0])

    layer = model.model.model[target_layer]
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    try:
        preds = model.model(inp)

        # ---- YOLO-safe scalar loss ----
        if isinstance(preds, dict):
            loss = sum(v.sum() for v in preds.values() if torch.is_tensor(v))
        elif isinstance(preds, (list, tuple)):
            loss = sum(p.sum() for p in preds if torch.is_tensor(p))
        else:
            loss = preds.sum()


        model.model.zero_grad()
        loss.backward()

        A = activations[0]
        G = gradients[0]

        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * A).sum(dim=1))[0]

        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cam = (cam * 255).astype(np.uint8)

        overlay = img.copy()

        # ---------- BOX-RESTRICTED GRAD-CAM ----------
        overlay = img.copy()

        for b in boxes:
            mask = np.zeros(cam.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [b["points"]], 255)

            cam_norm = cv2.normalize(cam, None, 0, 255, cv2.NORM_MINMAX)
            heat = cv2.applyColorMap(cam_norm, cv2.COLORMAP_JET)
            heat = cv2.bitwise_and(heat, heat, mask=mask)
            overlay = cv2.addWeighted(overlay, 0.90, heat, 0.25, 0)


        return overlay


    finally:
        h1.remove()
        h2.remove()
        model.model.train(was_training)










