import argparse

from database.Database import Database
from processes.Camera import Camera


class CameraEngine:
    db = None

    def __init__(self):
        if self.db is None:
            self.db = Database()

    @staticmethod
    def do_work():
        None
        # # main image to save
        # image_path = os.path.join(os.getcwd(), "sharp.png")
        #
        # # license plate image to save
        # image_to_scan, detections = recon.scan_image(image_path)
        #
        # # OCR result to save (make it switch)
        # recon.ocr(image_to_scan, detections, 0.8, easy.ocr_with_easy_ocr)
        # recon.ocr(image_to_scan, detections, 0.8, k.ocr_with_keras_ocr)
        # recon.ocr(image_to_scan, detections, 0.8, pyt.ocr_with_pytesseract)

    def create_camera(self, folder_path):
        camera = Camera(folder_path)
        self.db.insert_camera(camera)
        camera = self.db.get_camera(camera)
        return camera

    def delete_camera(self, folder_path):
        camera = Camera(folder_path)
        self.db.delete_camera(camera)

    def fetch_camera(self, folder_path):
        camera = Camera(folder_path)
        camera = self.db.get_camera(camera)
        return camera

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This engine is responsible for camera work")
    parser.add_argument('--method', type=str, help='Create/Delete camera')
    parser.add_argument('--folder_path', type=str, help='Location of camera folder')

    args = parser.parse_args()

    engine = CameraEngine()
    if args.method is None or args.folder_path is None:
        parser.print_help()

    if args.method.lower() == 'delete':
        a = engine.fetch_camera(args.folder_path)
    elif args.method.lower() == 'create':
        engine.create_camera(args.folder_path)
    else:
        parser.print_help()
