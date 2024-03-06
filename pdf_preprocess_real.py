import io
from PIL import Image
import fitz

# convert pdf to png, document answering uses images not pdfs
doc = fitz.open('13focom2e.pdf')
zoom = 1.5  # Adjust zoom factor if needed
mat = fitz.Matrix(zoom, zoom)
pixmapz = []
for page in doc:
    pixmapz.append(page.get_pixmap(matrix=mat).tobytes())
pixmapz.pop(0)
# Calculate image dimensions after cropping
num_pixels_to_remove_vertical = int(150 * zoom)
num_pixels_to_remove_horizontal = int(100 * zoom)

sample_image = Image.open(io.BytesIO(pixmapz[0]))

max_image_width = sample_image.width - 2 * num_pixels_to_remove_horizontal
max_image_height = sample_image.height - 2 * num_pixels_to_remove_vertical
responses = []
for i, pixmap_bytes in enumerate(pixmapz):
    image = Image.open(io.BytesIO(pixmap_bytes))
    image = image.crop((num_pixels_to_remove_horizontal, num_pixels_to_remove_vertical,
                        image.width - num_pixels_to_remove_horizontal,
                        image.height - num_pixels_to_remove_vertical))
    image.save(rf'image_base\{i}.png')

