import pytesseract as pyt
from PIL import Image


class PyTesseractOCR:

    @staticmethod
    def ocr_with_pytesseract(region):
        img = Image.fromarray(region, 'RGB')
        text = pyt.image_to_string(img)
        return text, region
