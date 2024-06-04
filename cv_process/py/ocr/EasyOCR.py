import easyocr as eo


class EasyOCR:

    @staticmethod
    def ocr_with_easy_ocr(region):
        reader = eo.Reader(['en'])
        text = reader.readtext(region)[0][1]
        return text
