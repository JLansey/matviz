import fitz  # PyMuPDF
import io
import os
from PIL import Image
import shutil


def extract_images_from_pdf(input_pdf_path: str, output_folder: str) -> str:
    """
    Extracts all images from the given PDF and saves them in the specified output folder.
    Returns the path to a zipped folder containing all extracted images.
    """
    doc = fitz.open(input_pdf_path)
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_path = os.path.join(output_folder, f"page{page_index + 1}_img{img_index + 1}.{image_ext}")
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            image_paths.append(image_path)

    doc.close()

    # Zip the folder for easier distribution
    zip_path = shutil.make_archive(output_folder, 'zip', output_folder)
    return zip_path


def compress_pdf_images(input_pdf_path: str, output_pdf_path: str, dpi: int = 150, quality: int = 40) -> str:
    """
    Compresses all images in the PDF by rendering pages as compressed images.
    Returns the path to the compressed PDF.

    dpi: controls the resolution of the page rendering.
    quality: JPEG quality setting (0-100).
    """
    doc = fitz.open(input_pdf_path)
    compressed_pdf = fitz.open()

    for page_index in range(len(doc)):
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=quality)
        img_stream = img_byte_arr.getvalue()

        new_page = compressed_pdf.new_page(width=page.rect.width, height=page.rect.height)
        rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
        new_page.insert_image(rect, stream=img_stream)

    compressed_pdf.save(output_pdf_path)
    compressed_pdf.close()
    doc.close()

    return output_pdf_path
