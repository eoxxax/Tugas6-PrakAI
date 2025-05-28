# Nama Program : Tugas6
# Nama         : Siti Nailah Eko Putri Alawiyah
# NPM          : 140810230059
# Tanggal Buat : 28 Mei 2025
# Deskripsi    : Program color picker berdasarkan warna dominan

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import io
import cv2
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="Dominant Color Picker", layout="wide")
st.title("Dominant Color Picker dari Gambar")

# Fungsi generate PDF
def generate_palette_pdf(colors, hex_colors, image):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Times", style="B", size=18)  
    pdf.cell(200, 10, txt="Dominant Color Picker dari Gambar", ln=True, align="C")

    # Simpan gambar sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        image.save(tmpfile.name)
        image_path = tmpfile.name

    # Ukuran gambar (dalam cm)
    img_width = pdf.w - 20  # misal padding 10 kiri dan kanan
    orig_width, orig_height = image.size
    ratio = img_width / orig_width
    img_height = orig_height * ratio

    # Tambah gambar
    pdf.image(image_path, x=10, y=25, w=img_width)

    # Hitung Y bawah gambar
    y_after_img = 25 + img_height + 10  # +10 buat spasi
    pdf.set_y(y_after_img)

    # Set posisi X agar tabel berada di tengah
    cell_widths = [15, 30, 50, 50]
    total_table_width = sum(cell_widths)
    x_start = (pdf.w - total_table_width) / 2
    pdf.set_x(x_start)

    # Header tabel
    pdf.set_font("Times", style="B", size=11)
    pdf.cell(cell_widths[0], 10, "No", 1, 0, "C")
    pdf.cell(cell_widths[1], 10, "Warna", 1, 0, "C")
    pdf.cell(cell_widths[2], 10, "RGB", 1, 0, "C")
    pdf.cell(cell_widths[3], 10, "HEX", 1, 1, "C")

    # Isi tabel
    pdf.set_font("Times", size=11)
    for idx, (rgb, hex_code) in enumerate(zip(colors, hex_colors), start=1):
        rgb_int = tuple(int(x) for x in rgb)
        pdf.set_fill_color(*rgb_int)

        pdf.set_x(x_start)
        pdf.cell(cell_widths[0], 10, str(idx), 1, 0, "C")
        pdf.cell(cell_widths[1], 10, "", 1, 0, fill=True)  
        pdf.cell(cell_widths[2], 10, str(rgb_int), 1, 0, "C")
        pdf.cell(cell_widths[3], 10, hex_code, 1, 1, "C")



    os.remove(image_path)
    return pdf.output(dest='S').encode('latin1')

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", width=850)

    # Ubah ke numpy dan buang alpha channel kalau ada
    image_np = np.array(image)
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    # Resize dan reshape
    resized_img = cv2.resize(image_np, (100, 100))
    reshaped_img = resized_img.reshape((-1, 3))

    # KMeans
    k = st.slider("Jumlah warna dominan", 1, 10, 5)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reshaped_img)
    colors = kmeans.cluster_centers_.astype(int)
    hex_colors = ['#%02x%02x%02x' % tuple(color) for color in colors]

    st.markdown("## Palet Warna Dominan")
    cols = st.columns(k)
    for i, col in enumerate(cols):
        with col:
            st.color_picker(f"Warna {i+1}", value=hex_colors[i])
            rgb_tuple = tuple(int(x) for x in colors[i])
            st.markdown(f"**RGB:** {rgb_tuple}")
            st.markdown(f"**HEX:** `{hex_colors[i]}`")

    # Tabel warna
    df_colors = pd.DataFrame({
        "Warna": [f"Warna {i+1}" for i in range(k)],
        "RGB": [tuple(int(x) for x in color) for color in colors],
        "Hex": hex_colors
    })
    st.markdown("### Tabel Warna")
    st.dataframe(df_colors, use_container_width=True)

    # Tombol download PDF
    pdf_bytes = generate_palette_pdf(colors, hex_colors, image)
    st.download_button(
        label="Unduh Palet sebagai PDF",
        data=pdf_bytes,
        file_name="palet_warna.pdf",
        mime="application/pdf"
    )

else:
    st.info("Silakan upload gambar terlebih dahulu.")
