import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from groq import Groq
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Damage Detection App", layout="wide")

st.markdown(
    "<h1 style='text-align: center; font-size: 48px; font-weight: bold;'>"
    "AI-Based Goods Damage Detection and Email Reporting Application</h1>",
    unsafe_allow_html=True,
)

# 🔐 API Key
groq_api_key = "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"
client = Groq(api_key=groq_api_key)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # path to trained model

model = load_model()

# ------------------- DAMAGE SEVERITY LOGIC -------------------
# ------------------- DAMAGE SEVERITY LOGIC -------------------
def calculate_severity(result):
    """
    Calculates overall damage severity ratio from YOLO results.
    Works even if multiple damaged regions exist (sums all).
    """
    boxes = result.boxes.xyxy.cpu().numpy()   # [x1,y1,x2,y2]
    classes = result.boxes.cls.cpu().numpy() # class ids

    package_area = None
    damage_area = 0

    for box, cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if cls_id == 1:  # package
            package_area = area
        elif cls_id == 0:  # damaged
            damage_area += area  # sum all damages

    severity = "No Package Detected"
    ratio = 0
    if package_area and damage_area > 0:
        ratio = damage_area / package_area
        if ratio < 0.1:
            severity = "Low"
        elif ratio < 0.3:
            severity = "Medium"
        else:
            severity = "High"

    return severity, ratio


# ------------------- EMAIL GENERATION -------------------
def generate_email(damaged, non_damaged, product_type, shipment_method, packaging_type, fragile_label, transit_days, template_type):
    subject = f"Inspection Results – {product_type} Shipment"
    if template_type == "Internal Team":
        prompt = f"""
Subject: {subject}
Dear Team,

The recent shipment of **{product_type}** has been inspected.

Shipment Details:
- Shipment Method: {shipment_method}
- Packaging Used: {packaging_type}
- Fragile Label Applied: {"Yes" if fragile_label else "No"}
- Time in Transit: {transit_days} days

Inspection Summary:
- Damaged Items: {damaged}
- Non-Damaged Items: {non_damaged}

This is an internal inspection report for documentation purposes.

Best regards,  
Quality Control Team
"""
    elif template_type == "Vendor Notification":
        prompt = f"""
Subject: Damage Found in Shipment – {product_type}
Dear Vendor,

We have completed the inspection of the recent shipment of **{product_type}**.

Details:
- Shipment Method: {shipment_method}
- Packaging Type: {packaging_type}
- Fragile Label: {"Yes" if fragile_label else "No"}
- Transit Time: {transit_days} days

Inspection Results:
- Damaged Packages: {damaged}
- Non-Damaged Packages: {non_damaged}

Please review the shipment and ensure packaging compliance in future dispatches.

Regards,  
Quality Control Department
"""
    elif template_type == "Customer Notification":
        prompt = f"""
Subject: Update on Your Shipment – {product_type}
Dear Customer,

We’re writing to update you on your recent shipment of **{product_type}**.

Shipment Method: {shipment_method}  
Packaging: {packaging_type}  
Fragile Label: {"Yes" if fragile_label else "No"}  
Transit Duration: {transit_days} days  

Inspection Outcome:
- Damaged Units: {damaged}
- Non-Damaged Units: {non_damaged}

If any action is needed on your order, our support team will reach out to you shortly.

Best regards,  
Customer Support  
"""
    else:
        return "⚠️ Invalid email template selected."

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "system", "content": "You are a professional quality control manager writing a formal email."},
                      {"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating email: {e}"

# ------------------- SESSION STATE -------------------
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "results" not in st.session_state:
    st.session_state.results = []
if "counts" not in st.session_state:
    st.session_state.counts = {"damaged": 0, "non_damaged": 0}
if "email_text" not in st.session_state:
    st.session_state.email_text = ""

# ------------------- TABS -------------------
tab1, tab2, tab3 = st.tabs(["🔧 Inputs & Upload", "🖼 Detection & Results", "📧 Generated Email"])

# ------------------- TAB 1 -------------------
with tab1:
    st.header("🔧 Enter Shipment Details & Upload Images")

    col1, col2 = st.columns([1, 1])

    with col1:
        product_type = st.selectbox("Product Type", ["Electronics", "Furniture", "Clothing", "Food", "Other"])
        shipment_method = st.selectbox("Shipment Method", ["Air", "Sea", "Land", "Courier"])
        packaging_type = st.selectbox("Packaging Type", ["Bubble Wrap", "Foam", "Cardboard Box", "Wooden Crate", "None"])
        fragile_label = st.checkbox("Fragile Label Applied?")
        transit_days = st.number_input("Transit Time (Days)", min_value=0, step=1)
        template_type = st.selectbox("Email Template", ["Internal Team", "Vendor Notification", "Customer Notification"])

    with col2:
        uploaded_files = st.file_uploader("📥 Upload Product Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"{len(uploaded_files)} image(s) uploaded successfully.")
            st.markdown("### 📸 Uploaded Preview")
            cols = st.columns(4)
            for i, file in enumerate(uploaded_files):
                img = Image.open(file)
                cols[i % 4].image(img, caption=file.name, use_container_width=True)

# ------------------- TAB 2 -------------------
with tab2:
    st.header("🖼 Image Damage Detection")

    if not st.session_state.uploaded_files:
        st.warning("⚠️ No images uploaded. Please go to 'Inputs & Upload' tab first.")
    else:
        if st.button("🚀 Run Damage Detection"):
            results = []
            damaged = 0
            non_damaged = 0

            for file in st.session_state.uploaded_files:
                img = Image.open(file).convert("RGB")
                img_np = np.array(img)

                # Run YOLO prediction
                yolo_results = model.predict(img_np, conf=0.25, imgsz=640, verbose=False)
                annotated_img = yolo_results[0].plot()  # annotated image

                # Severity calculation (overall)
                severity, ratio = calculate_severity(yolo_results[0])

                # Decide classification
                if severity == "No Package Detected" or ratio == 0:
                    label = "Not Damaged"
                    non_damaged += 1
                else:
                    label = f"Damaged ({severity})"
                    damaged += 1

                results.append((file.name, annotated_img, label, severity, ratio))

            st.session_state.results = results
            st.session_state.counts = {"damaged": damaged, "non_damaged": non_damaged}
            st.success("✅ Damage detection complete!")

        if st.session_state.results:
            st.subheader("📊 Summary")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("✅ Not Damaged", st.session_state.counts['non_damaged'])
                st.metric("❌ Damaged", st.session_state.counts['damaged'])
            with col2:
                labels = ['Damaged', 'Not Damaged']
                sizes = [st.session_state.counts["damaged"], st.session_state.counts["non_damaged"]]
                colors = ['#FF4B4B', '#4CAF50']
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

            st.subheader("🔍 Classification Results")
            cols = st.columns(2)
            for i, (filename, annotated_img, label, severity, ratio) in enumerate(st.session_state.results):
                caption = f"{filename} - {label} | Severity: {severity} ({ratio:.2f})"
                cols[i % 2].image(annotated_img, caption=caption, use_container_width=True)

# ------------------- TAB 3 -------------------
with tab3:
    st.header("📧 Auto-Generated Email")
    if not st.session_state.results:
        st.warning("⚠️ No inspection results found. Please run detection in Tab 2.")
    else:
        if st.button("✉️ Generate Email"):
            email = generate_email(
                st.session_state.counts["damaged"],
                st.session_state.counts["non_damaged"],
                product_type,
                shipment_method,
                packaging_type,
                fragile_label,
                transit_days,
                template_type
            )
            st.session_state.email_text = email
            st.success("📨 Email generated!")
        if st.session_state.email_text:
            st.subheader("📄 Email Content")
            st.code(st.session_state.email_text, language="markdown")