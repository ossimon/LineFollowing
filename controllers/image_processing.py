from PIL import Image
import numpy as np

def bytes_to_image(image_bytes, width, height):
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = image.reshape((height, width, 4))
    image = image[:, :, [2, 1, 0, 3]]
    return image

def save_image(image, filename):
    image = Image.fromarray(image, mode='RGBA')
    image.save("../../images/" + filename)