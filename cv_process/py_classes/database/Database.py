import os
import xml.etree.ElementTree as ET

import psycopg2 as pg


class Database:
    connection = None

    def __init__(self):
        self.connection = self.__establish_connection()

    @staticmethod
    def __properties():
        base = ET.parse(os.path.join(os.getcwd(), "cv_process", "py_classes", "database", "db_connection.xml"))
        root = base.getroot()

        properties = {
            "database": root.find("database").text,
            "host": root.find("host").text,
            "user": root.find("user").text,
            "password": root.find("password").text,
            "port": root.find("port").text
        }
        return properties

    @staticmethod
    def __establish_connection():
        conn = pg.connect(database=Database.__properties()["database"],
                          host=Database.__properties()["host"],
                          user=Database.__properties()["user"],
                          password=Database.__properties()["password"],
                          port=Database.__properties()["port"])
        return conn

    def get_camera(self, camera):
        sql = "select * from thesis.camera where name = '{}' and folder_path = '{}'".format(
            camera.name, camera.folder_path
        )
        existing_camera = None
        with self.connection as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                existing_camera = cursor.fetchone()
        return existing_camera

    def insert_placeholder_image(self, image_path):
        sql = ("INSERT INTO thesis.image (license_plate_number, license_plate_image_path, full_image_path) "
               + "VALUES ('-', '-', '{}') RETURNING id".format(image_path))
        with self.connection as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                last_id = cursor.fetchone()[0]
                conn.commit()
                return last_id

    def update_image(self, image, id_to_use):
        sql = (
                "UPDATE thesis.image SET license_plate_number = '{}', license_plate_image_path = '{}', full_image_path = '{}', camera_id = '{}' where id = '{}'".format(
                    image.license_plate_number, image.license_plate_image_path, image.full_image_path, image.camera_id, id_to_use
                ))
        print(sql)
        with self.connection as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                conn.commit()

    def image_exists(self, image_path):
        sql = ("SELECT * FROM thesis.image WHERE full_image_path = '{}'".format(image_path))
        with self.connection as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                image = cursor.fetchone()
        return image
