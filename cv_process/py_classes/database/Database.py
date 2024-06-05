import os
import xml.etree.ElementTree as ET

import psycopg2 as pg


class Database:
    connection = None

    def __init__(self):
        self.connection = self.__establish_connection()

    @staticmethod
    def __properties():
        base = ET.parse(os.path.join("py_classes", "database", "db_connection.xml"))
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

    def insert_camera(self, camera):
        sql = "INSERT INTO thesis.camera (name, folder_path, image_format) VALUES ('{}', '{}', '{}')".format(
            camera.name, camera.folder_path, camera.image_format
        )
        with self.connection as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                conn.commit()

    def delete_camera(self, camera):
        sql = "DELETE from thesis.camera where name = '{}' and folder_path = '{}'".format(
            camera.name, camera.folder_path
        )
        with self.connection as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                conn.commit()

    def insert_image(self, image):
        sql = ("INSERT INTO thesis.image (license_plate_number, license_plate_image_path, full_image_path, camera_id) "
               + "VALUES ('{}', '{}', '{}', '{}')".format(
                    image.license_plate_number, image.license_plate_image_path, image.full_image_path, image.camera_id
                ))
        with self.connection as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                conn.commit()
