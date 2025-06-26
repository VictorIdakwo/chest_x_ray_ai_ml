import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import io
from PIL import Image

# ✅ Set Streamlit config (must be first)
st.set_page_config(page_title="🩺 Chest X-ray Pneumonia Detection", layout="centered")

MODEL_PATH = "chest_xray_model.h5"  # Or use .keras if preferred

# Load model
@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    dummy_input = np.zeros((1, 150, 150, 3))  # ensure model is "built"
    model.predict(dummy_input)
    return model

model = load_model()

# Grad-CAM function
def get_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Threshold-based interpretation
def interpret_confidence(pred_score):
    if pred_score >= 0.90:
        return "🔴 High-confidence Pneumonia", "This case likely shows pneumonia. Clinical confirmation still required."
    elif 0.70 <= pred_score < 0.90:
        return "🟠 Possible Pneumonia", "Findings suggest pneumonia; please confirm with a radiologist or further tests."
    else:
        return "🟢 Likely Normal", "Low confidence for pneumonia. Human interpretation still advised."

# ───────────────────────────────────────────────────────────────
# Streamlit UI
st.image("static/eHA-logo-blue_320x132.png", width=200)
st.title("🩺 Chest X-ray Pneumonia Detection (AI/ML Prototype)")
st.caption("**Test application for eHealth Africa Clinic - AI-assisted Chest X-ray Interpretation.**")
st.caption("Models trained on open-source dataset: [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)")

st.write("Upload a chest X-ray to get AI-assisted diagnosis and Grad-CAM visualization.")

uploaded_file = st.file_uploader("📤 Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.warning("⚠️ File too large. Please upload an image under 5MB.")
        else:
            # Show uploaded image
            st.image(uploaded_file, caption="🖼️ Uploaded X-ray", use_container_width=True)

            # Preprocess image
            img = load_img(uploaded_file, target_size=(150, 150))
            img_array = img_to_array(img) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            # Make prediction
            pred_score = float(model.predict(img_batch)[0][0])
            predicted_class = "Pneumonia" if pred_score >= 0.5 else "Normal"
            diagnosis_label, comment = interpret_confidence(pred_score)

            st.markdown(f"### 🧠 Diagnosis: **{diagnosis_label}**")
            st.progress(float(pred_score))
            st.markdown(f"**Confidence Score:** `{pred_score * 100:.2f}%`")
            st.markdown(f"**Predicted Class:** `{predicted_class}`")
            st.info(comment)

            # Grad-CAM heatmap
            heatmap = get_gradcam_heatmap(img_batch, model)
            img_cv = cv2.cvtColor(np.uint8(img_array * 255), cv2.COLOR_RGB2BGR)
            heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

            st.image(superimposed_img, caption="🔥 Grad-CAM Anomaly Map", channels="BGR", use_container_width=True)

            # Optional: download button
            result_image = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            st.download_button("📥 Download Grad-CAM Image", buf.getvalue(), file_name="gradcam_result.png", mime="image/png")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
