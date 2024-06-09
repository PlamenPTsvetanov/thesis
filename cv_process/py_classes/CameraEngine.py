from database.Database import Database
from objects.Camera import Camera


class CameraEngine:
    db = None

    def __init__(self):
        if self.db is None:
            self.db = Database()

    def fetch_camera(self, folder_path):
        camera = Camera(folder_path)
        camera = self.db.get_camera(camera)
        return camera
