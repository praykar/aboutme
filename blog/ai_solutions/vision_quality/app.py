import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io

st.set_page_config(page_title="Vision Quality Control Demo", layout="wide")

st.title("🔍 Vision Quality Control Demo")
st.markdown(
    """A minimal computer vision quality control demo showing image inspection with 
    dummy pass/fail classification based on brightness, blur detection, and contrast."""
)

# Sidebar
st.sidebar.header("QC Parameters")
brightness_threshold = st.sidebar.slider("Brightness Threshold", 50, 200, 100)
blur_threshold = st.sidebar.slider("Blur Threshold (Variance)", 50, 500, 150)
contrast_threshold = st.sidebar.slider("Contrast Threshold", 20, 100, 40)

# Helper functions
def calculate_brightness(image):
    """Calculate average brightness"""
    img_array = np.array(image.convert('L'))
    return np.mean(img_array)

def calculate_blur(image):
    """Calculate blur using Laplacian variance"""
    img_array = np.array(image.convert('L'))
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # Simple convolution
    padded = np.pad(img_array, 1, mode='edge')
    result = np.zeros_like(img_array, dtype=float)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            result[i, j] = np.sum(padded[i:i+3, j:j+3] * laplacian)
    return np.var(result)

def calculate_contrast(image):
    """Calculate image contrast (std deviation)"""
    img_array = np.array(image.convert('L'))
    return np.std(img_array)

def perform_qc(image, brightness_th, blur_th, contrast_th):
    """Perform quality control checks"""
    brightness = calculate_brightness(image)
    blur = calculate_blur(image)
    contrast = calculate_contrast(image)
    
    checks = {
        'brightness': brightness,
        'blur': blur,
        'contrast': contrast,
        'brightness_pass': brightness >= brightness_th * 0.5 and brightness <= brightness_th * 1.5,
        'blur_pass': blur >= blur_th,
        'contrast_pass': contrast >= contrast_th
    }
    
    checks['overall_pass'] = checks['brightness_pass'] and checks['blur_pass'] and checks['contrast_pass']
    
    return checks

# Main interface
st.subheader("📷 Image Upload")

# File uploader
uploaded_file = st.file_uploader("Upload an image for quality inspection", type=['png', 'jpg', 'jpeg'])

# Sample image generator
if st.button("Generate Sample Image"):
    # Create a synthetic sample image
    img = Image.new('RGB', (400, 300), color=(150, 150, 150))
    # Add some random noise
    img_array = np.array(img)
    noise = np.random.normal(0, 25, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to bytes for display
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    st.session_state['sample_image'] = buf

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    display_qc_results(image)
elif 'sample_image' in st.session_state:
    image = Image.open(st.session_state['sample_image'])
    display_qc_results(image)
else:
    st.info("Please upload an image or generate a sample image to begin quality inspection")

def display_qc_results(image):
    """Display image and QC results"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Quality Control Results")
        
        # Perform QC
        results = perform_qc(image, brightness_threshold, blur_threshold, contrast_threshold)
        
        # Display overall result
        if results['overall_pass']:
            st.success("✅ PASS: Image meets quality standards")
        else:
            st.error("❌ FAIL: Image does not meet quality standards")
        
        # Display metrics
        st.markdown("### Quality Metrics")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            status = "✅" if results['brightness_pass'] else "❌"
            st.metric("Brightness", f"{results['brightness']:.1f}", delta=status)
        
        with col_b:
            status = "✅" if results['blur_pass'] else "❌"
            st.metric("Sharpness", f"{results['blur']:.1f}", delta=status)
        
        with col_c:
            status = "✅" if results['contrast_pass'] else "❌"
            st.metric("Contrast", f"{results['contrast']:.1f}", delta=status)
        
        # Detailed checks
        st.markdown("### Detailed Checks")
        checks_data = [
            ("Brightness Range", "Pass" if results['brightness_pass'] else "Fail", 
             f"{results['brightness']:.1f} (Target: {brightness_threshold * 0.5:.1f}-{brightness_threshold * 1.5:.1f})"),
            ("Blur Detection", "Pass" if results['blur_pass'] else "Fail",
             f"{results['blur']:.1f} (Minimum: {blur_threshold})"),
            ("Contrast Level", "Pass" if results['contrast_pass'] else "Fail",
             f"{results['contrast']:.1f} (Minimum: {contrast_threshold})")
        ]
        
        for check_name, status, value in checks_data:
            col_x, col_y, col_z = st.columns([2, 1, 3])
            col_x.write(check_name)
            if status == "Pass":
                col_y.markdown(":green[Pass]")
            else:
                col_y.markdown(":red[Fail]")
            col_z.write(value)

# Call function if image exists
if uploaded_file is not None:
    pass  # Already handled above
elif 'sample_image' in st.session_state:
    pass  # Already handled above

# Footer
st.markdown("---")
st.caption(
    "This is a simplified vision QC demo using basic image metrics. "
    "Production systems should use deep learning models (CNN, YOLO, etc.) for defect detection, "
    "classification, and segmentation with training on domain-specific datasets."
)
