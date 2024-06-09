class CustomImage:
    id = None
    license_plate_number = None
    license_plate_image_path = None
    full_image_path = None
    camera_id = None

    def __init__(self, license_plate_number, license_plate_image_path, full_image_path, camera_id, id=None):
        self.id = id
        self.license_plate_number = license_plate_number
        self.license_plate_image_path = license_plate_image_path
        self.full_image_path = full_image_path
        self.camera_id = camera_id
