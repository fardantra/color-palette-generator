import io

import numpy as np
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans

st.set_page_config(page_title="Color Palette Generator", layout="centered")

st.markdown(
    """
    <style>
    .title-text {
        text-align: center;
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 800;
        margin-bottom: 30px;
    }
    .palette-container {
        display: flex;
        width: 100%;
        height: 160px;
        border-radius: 12px;
        overflow: hidden;
        margin-top: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    .color-box {
        flex: 1;
        height: 100%;
        transition: flex 0.4s ease, filter 0.3s ease;
    }
    .color-box:hover {
        flex: 1.5;
        filter: brightness(1.1);
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 class='title-text'>Color Palette Generator</h1>", unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Unggah gambar untuk dianalisis (JPG, PNG)", type=["jpg", "jpeg", "png"]
)


def extract_colors_kmeans(img, n_colors=8):

    img_array = np.array(img)

    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    pixels = img_array.reshape(-1, 3)

    if len(pixels) > 15000:
        idx = np.random.choice(len(pixels), 15000, replace=False)
        pixels_sample = pixels[idx]
    else:
        pixels_sample = pixels

    kmeans = KMeans(n_clusters=n_colors, n_init="auto", random_state=42)
    kmeans.fit(pixels_sample)

    colors = kmeans.cluster_centers_.astype(int)

    colors = colors[np.argsort(np.sum(colors, axis=1))]

    return colors


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(image, use_container_width=True, caption="Gambar Sumber")

    with st.spinner("Menganalisis harmoni warna dengan AI K-Means..."):
        try:
            colors = extract_colors_kmeans(image, n_colors=8)

            st.markdown("### Color Palette:")

            # Build Palette UI (Pure color bars)
            palette_html = '<div class="palette-container">'
            for rgb in colors:
                hex_color = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2]).upper()
                palette_html += f'<div class="color-box" style="background-color: {hex_color};"></div>'
            palette_html += "</div>"

            st.markdown(palette_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("Silakan unggah gambar untuk memulai.")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #95a5a6; font-size: 0.8rem;'>Fardan Fadhilah Andicha Putra - 140810240084</p>",
    unsafe_allow_html=True,
)
