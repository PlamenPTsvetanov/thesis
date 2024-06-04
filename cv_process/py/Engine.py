from RecognitionSetup import Setup
from Recognition import Recognition
from Training import Training
from ocr.EasyOCR import EasyOCR as easy
from ocr.KerasOCR import KerasOCR as k
from ocr.PyTesseractOCR import PyTesseractOCR as pyt

import os


class Engine:
    @staticmethod
    def run():
        meta = Setup()
        meta.setup()

    @staticmethod
    def train():
        Engine.run()
        train = Training()
        train.train_model()

    @staticmethod
    def test():
        recon = Recognition()
        image_to_scan, detections = recon.scan_image(os.path.join(os.getcwd(), "sharp.png"))

        recon.ocr(image_to_scan, detections, 0.8, easy.ocr_with_easy_ocr)
        recon.ocr(image_to_scan, detections, 0.8, k.ocr_with_keras_ocr)
        recon.ocr(image_to_scan, detections, 0.8, pyt.ocr_with_pytesseract)


Engine.test()
