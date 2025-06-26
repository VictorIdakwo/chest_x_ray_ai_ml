import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# ✅ Set Streamlit config (must be first)
st.set_page_config(page_title="🩺 Chest X-ray Pneumonia Detection")

MODEL_PATH = "chest_xray_model.h5"  # Or use .keras if preferred

# Load model
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

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

# Streamlit UI
st.title("🩺 Chest X-ray Pneumonia Detection")
st.write("Upload a chest X-ray to get AI-assisted diagnosis and Grad-CAM visualization.")

uploaded_file = st.file_uploader("📤 Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Show uploaded image
        st.image(uploaded_file, caption="🖼️ Uploaded X-ray", use_container_width=True)

        # Preprocess image
        img = load_img(uploaded_file, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Make prediction
        pred_score = float(model.predict(img_batch)[0][0])
        diagnosis_label, comment = interpret_confidence(pred_score)

        st.markdown(f"### 🧠 Diagnosis: **{diagnosis_label}**")
        st.progress(pred_score)
        st.markdown(f"**Confidence Score:** `{pred_score * 100:.2f}%`")
        st.info(comment)

        # Grad-CAM heatmap
        heatmap = get_gradcam_heatmap(img_batch, model)
        img_cv = cv2.cvtColor(np.uint8(img_array * 255), cv2.COLOR_RGB2BGR)
        heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

        st.image(superimposed_img, caption="🔥 Grad-CAM Anomaly Map", channels="BGR", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
