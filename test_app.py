import json
import numpy as np
import tensorflow as tf
import gradio as gr

print("Loading Agrilens Pro models...")
gate_model = tf.keras.models.load_model("agrilens_pro_gate.keras")
clf_model = tf.keras.models.load_model("agrilens_pro_classifier.keras")
class_names = {int(k): v for k, v in json.load(open("class_names.json")).items()}

GATE_THR = 0.80
CLF_THR = 0.60

def check_image_quality(image):
    gray = np.mean(image, axis=2)
    brightness = float(np.mean(gray))
    # Simple edge sharpness check (variance of pixel differences)
    sharpness = float(np.var(np.diff(gray, axis=0)) + np.var(np.diff(gray, axis=1)))

    if brightness < 35:
        return "⚠️ IMAGE TOO DARK\nPlease move to brighter, natural daylight and retake the photo."
    if brightness > 235:
        return "⚠️ IMAGE OVEREXPOSED\nDirect glare is washing out leaf details. Shade the leaf and retake."
    if sharpness < 60:
        return "⚠️ IMAGE BLURRY\nHold the camera steady and tap the leaf to focus before capturing."
    return None

def get_crops(image):
    h, w, _ = image.shape
    ch, cw = int(h * 0.65), int(w * 0.65)
    cy, cx = (h - ch) // 2, (w - cw) // 2
    return [
        image,                        # 0: Full image
        image[cy:cy+ch, cx:cx+cw],    # 1: Center zoom (65%)
        image[0:ch, 0:cw],            # 2: Top-left
        image[0:ch, w-cw:w],          # 3: Top-right
        image[h-ch:h, 0:cw],          # 4: Bottom-left
        image[h-ch:h, w-cw:w],        # 5: Bottom-right
    ]

def test_leaf(image):
    if image is None:
        return "Please upload a tomato leaf image to begin.", {}

    # 1. Pre-check lighting and focus
    quality_issue = check_image_quality(image)
    if quality_issue:
        return quality_issue, {}

    # 2. Evaluate full image + 5 zoomed crops in one fast batch
    crops = get_crops(image)
    batch = tf.stack([tf.cast(tf.image.resize(c, (224, 224), method="bilinear"), tf.float32) for c in crops])

    gate_scores = gate_model.predict(batch, verbose=0).ravel()
    full_gate = float(gate_scores[0])
    best_idx = int(np.argmax(gate_scores))
    p_tomato = float(gate_scores[best_idx])

    # Reject non-tomato images
    if p_tomato < GATE_THR:
        return (
            f"❌ NOT A TOMATO LEAF (Gate Score: {p_tomato:.1%})\n"
            f"• Please upload a clear photo of a single tomato leaf.\n"
            f"• Avoid photos of stems, soil, hands, or other plant species."
        ), {}

    # Detect wide canopy / too many leaves in frame
    high_crop_count = int(np.sum(gate_scores[1:] >= 0.85))
    if full_gate < 0.55 and high_crop_count >= 2:
        return (
            f"🌿 TOO MANY LEAVES IN FRAME (Full View Score: {full_gate:.1%})\n"
            f"• Multiple leaves or dense foliage detected.\n"
            f"• Please isolate ONE symptomatic leaf in the center and retake."
        ), {}

    # 3. Classify the best-framed view (prefer Full Image if it already passes Gate)
    chosen_idx = 0 if full_gate >= GATE_THR else best_idx
    chosen_gate = float(gate_scores[chosen_idx])
    best_crop = tf.expand_dims(batch[chosen_idx], axis=0)

    probs = clf_model.predict(best_crop, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    conf = float(probs[top_idx])
    disease = class_names[top_idx].replace("_", " ").title()

    top3_idx = np.argsort(probs)[::-1][:3]
    top3_scores = {class_names[int(i)].replace("_", " ").title(): float(probs[i]) for i in top3_idx}
    view_used = "Full Image" if chosen_idx == 0 else f"Auto-Zoom Crop #{chosen_idx}"

    if conf < CLF_THR:
        return (
            f"⚠️ UNCERTAIN DIAGNOSIS — PLEASE RETAKE ({conf:.1%} confidence)\n"
            f"• Tomato Leaf Verified: {chosen_gate:.1%} ({view_used})\n"
            f"• Closest Match: {disease}\n"
            f"• Tip: Frame ONE leaf flat against a plain background in good light."
        ), top3_scores

    return (
        f"✅ DIAGNOSIS COMPLETE ({view_used})\n"
        f"• Condition: {disease} ({conf:.1%} confidence)\n"
        f"• Tomato Leaf Verified: {chosen_gate:.1%}\n"
        f"• Note: If multiple leaves are affected, inspect individual leaves to confirm."
    ), top3_scores

demo = gr.Interface(
    fn=test_leaf,
    inputs=gr.Image(type="numpy", label="Upload Tomato Leaf Photo"),
    outputs=[
        gr.Textbox(label="Agrilens Pro Assessment", lines=5),
        gr.Label(label="Top 3 Condition Probabilities")
    ],
    title="Agrilens Pro — Tomato Leaf Disease Detector",
    description=(
        "📸 Photo Guidelines: Capture ONE leaf per photo • Hold camera 15–20 cm away • "
        "Use bright, indirect daylight • Keep the leaf flat and in focus."
    )
)

if __name__ == "__main__":
    demo.launch()
