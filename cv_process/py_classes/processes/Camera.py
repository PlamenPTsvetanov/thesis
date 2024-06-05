import os
import xml.etree.ElementTree as ET


class Camera:
    folder_path = None
    name = None
    image_format = None
    id = None
    def __init__(self, path):
        if self.folder_path is None:
            self.folder_path = path
        base = ET.parse(os.path.join(str(self.folder_path), "camera_config.xml"))
        self.name = base.getroot().find("name").text
        self.image_format = base.getroot().find("image_format").text
