import argparse
import glob
import os
import time
import uuid
from functools import partial

import schedule
from PIL import Image

from CameraEngine import CameraEngine
from RLDeblur import RLDeblur
from Recognition import Recognition
from database.Database import Database
from objects.CustomImage import CustomImage
from ocr.EasyOCR import EasyOCR as easy
from ocr.KerasOCR import KerasOCR as k
from ocr.PyTesseractOCR import PyTesseractOCR as pyt


class ImageEngine:
    def recognize(self, path, ocr_func):
        print("Starting recognition...")
        camera_engine = CameraEngine()

        camera = camera_engine.fetch_camera(folder_path=path)

        recon = Recognition()

        images = glob.glob(os.path.join(path, "*." + camera[6]))

        db = Database()

        for image_path in images:
            db_image = db.image_exists(image_path)
            if db_image is None:
                id_to_use = db.insert_placeholder_image(image_path)
            else:
                continue

            output_folder = os.path.join(path, str(uuid.uuid4()))
            os.mkdir(output_folder)

            img_format = camera[6]
            deblur = RLDeblur()

            deblurred_path = deblur.deblur(path=image_path, output_path=output_folder, format=img_format)

            image_to_scan, detections = recon.scan_image(deblurred_path)
            text, region = recon.ocr(image_to_scan, detections, 0.8, ocr_func)

            full_image = Image.open(image_path)
            full_image_path = os.path.join(output_folder, "full_image" + "." + img_format)
            full_image.save(full_image_path)

            license_plate_image = Image.fromarray(region)
            license_plate_image_path = os.path.join(output_folder, "license_plate_image" + "." + img_format)
            license_plate_image.save(license_plate_image_path)

            license_plate_number = text
            camera_id = camera[0]

            image = CustomImage(license_plate_number, license_plate_image_path, full_image_path, camera_id)
            db.update_image(image, id_to_use)

            os.remove(image_path)

    def cron_worker(self, path, func):
        schedule.every(10).seconds.do(partial(self.recognize, path=path, ocr_func=func))
        while True:
            print("Running cron...")
            schedule.run_pending()
            time.sleep(1)

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description="This engine is responsible for image processing")
        parser.add_argument('--ocr', type=str, help='Which OCR method to use - KerasOCR, PyTesseract, EasyOCR. '
                                                    'Default: EasyOCR', default="EasyOCR")
        parser.add_argument('--folder', type=str, help='Where to look for images')

        args = parser.parse_args()

        ie = ImageEngine()
        if args.folder is None:
            parser.print_help()
            return

        folder = args.folder

        if args.ocr.lower() == 'kerasocr':
            func = k.ocr_with_keras_ocr
        elif args.ocr.lower() == 'pytesseract':
            func = pyt.ocr_with_pytesseract
        else:
            func = easy.ocr_with_easy_ocr

        ie.cron_worker(folder, func)


if __name__ == "__main__":
    ImageEngine.main()
