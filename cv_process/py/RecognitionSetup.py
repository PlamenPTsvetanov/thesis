import wget
import os
import subprocess
import tarfile
import sys
from zipfile import ZipFile
import tensorflow as tf
import cv2
import numpy as np
import re
from google.protobuf import text_format

from matplotlib import pyplot as plt


class Setup:
    CUSTOM_MODEL_NAME = 'custom_ssd_mobilenet_v2'
    PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'

    paths = {
        'WORKSPACE_PATH': os.path.join('../workspace'),
        'SCRIPTS_PATH': os.path.join('../scripts'),
        'APIMODEL_PATH': os.path.join('../models'),
        'ANNOTATION_PATH': os.path.join('../workspace', 'annotations'),
        'IMAGE_PATH': os.path.join('../workspace', 'images'),
        'MODEL_PATH': os.path.join('../workspace', 'models'),
        'PRETRAINED_MODEL_PATH': os.path.join('../workspace', 'pre-trained-model'),
        'CHECKPOINT_PATH': os.path.join('../workspace', 'models', CUSTOM_MODEL_NAME),
        'OUTPUT_PATH': os.path.join('../workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
        'TFJS_PATH': os.path.join('../workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
        'TFLITE_PATH': os.path.join('../workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
        'PROTOC_PATH': os.path.join('../protoc')
    }

    files = {
        'PIPELINE_CONFIG': os.path.join('../workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }

    labels = [{'name': 'licence', 'id': 1}]

    def __create_paths(self):
        for path in self.paths.values():
            if not os.path.exists(path):
                os.mkdir(path)

    def __download_protoc(self):
        if not os.listdir(self.paths['PROTOC_PATH']):
            print('Downloading protoc...')
            url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.19.5/protoc-3.19.5-win64.zip"
            wget.download(url)
            os.replace(os.path.join(os.getcwd(), "protoc-3.19.5-win64.zip"),
                       os.path.join(self.paths['PROTOC_PATH'], "protoc-3.19.5-win64.zip"))
            with ZipFile(os.path.join(self.paths['PROTOC_PATH'], "protoc-3.19.5-win64.zip"), 'r') as protoc:
                protoc.extractall(self.paths['PROTOC_PATH'])
            try:
                os.chdir('models/research')
                subprocess.run(["protoc", "object_detection/protos/*.proto", "--python_out=."], check=True, shell=True)
                subprocess.run(["copy", "object_detection\\packages\\tf2\\setup.py", "setup.py"], check=True,
                               shell=True)
                subprocess.run([sys.executable, "setup.py", "build"], check=True)
                subprocess.run([sys.executable, "setup.py", "install"], check=True)
                os.chdir('slim')
                subprocess.run(["pip", "install", "-e", "."], check=True, shell=True)
                os.chdir(os.path.join("../..", "..", ".."))
            except subprocess.CalledProcessError as e:
                print("An exception occurred!" + str(e))

            os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(self.paths['PROTOC_PATH'], 'bin'))
            print(os.environ['PATH'])

    def __download_object_detection(self):
        if not os.path.exists(os.path.join(self.paths['APIMODEL_PATH'], 'research', 'object_detection')):
            subprocess.run(['git', 'clone', 'https://github.com/tensorflow/models', 'models'], check=True)

    def __download_pretrained_model(self):
        if not os.path.exists(self.paths['PRETRAINED_MODEL_PATH']):
            os.mkdir(self.paths['PRETRAINED_MODEL_PATH'])
            wget.download(self.PRETRAINED_MODEL_URL)
            os.replace(os.path.join(os.getcwd(), self.PRETRAINED_MODEL_NAME + '.tar.gz'),
                       os.path.join(self.paths['PRETRAINED_MODEL_PATH'], self.PRETRAINED_MODEL_NAME + '.tar.gz'))

            with tarfile.open(
                    os.path.join(self.paths['PRETRAINED_MODEL_PATH'], self.PRETRAINED_MODEL_NAME + '.tar.gz')) as file:
                file.extractall(path=os.path.join(self.paths['PRETRAINED_MODEL_PATH']))

    def __download_tf_record_creator(self):
        if not os.path.exists(self.files['TF_RECORD_SCRIPT']):
            subprocess.run(
                ['git', 'clone', 'https://github.com/nicknochnack/GenerateTFRecord', self.paths['SCRIPTS_PATH']],
                check=True)

    def __generate_tf_records(self):
        with open(self.files['LABELMAP'], 'w') as f:
            for label in self.labels:
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(label['name']))
                f.write('\tid:{}\n'.format(label['id']))
                f.write('}\n')
        os.system(sys.executable + " " + self.files['TF_RECORD_SCRIPT']
                  + ' -x ' + os.path.join(self.paths['IMAGE_PATH'], 'test')
                  + ' -l ' + self.files['LABELMAP']
                  + ' -o ' + os.path.join(self.paths['ANNOTATION_PATH'], 'test.record'))
        os.system(sys.executable + " " + self.files['TF_RECORD_SCRIPT']
                  + ' -x ' + os.path.join(self.paths['IMAGE_PATH'], 'train')
                  + ' -l ' + self.files['LABELMAP']
                  + ' -o ' + os.path.join(self.paths['ANNOTATION_PATH'], 'train.record'))
        os.system("copy "
                  + os.path.join(self.paths['PRETRAINED_MODEL_PATH'], self.PRETRAINED_MODEL_NAME, 'pipeline.config')
                  + " "
                  + os.path.join(self.paths['CHECKPOINT_PATH']))

    def __clear_text(self, text):
        text = ''.join(text)
        text = re.sub(r'[^a-zA-Z0-9]', '', text)
        return text

    def __common_ocr(self, image, detections, detection_threshold, ocr_function):
        scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
        boxes = detections['detection_boxes'][:len(scores)]

        width = image.shape[1]
        height = image.shape[0]

        for idx, box in enumerate(boxes):
            roi = box * [height, width, height, width]
            region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]

            text = ocr_function(region)
            plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            plt.show()
            return text, region

    def ocr(self, image_np_with_detections, detections, detection_threshold, ocr_function):
        image = image_np_with_detections
        text, region = self.__common_ocr(image, detections, detection_threshold, ocr_function)
        print(self.__clear_text(text))


    def setup(self):
        self.__create_paths()
        self.__download_protoc()
        self.__download_object_detection()
        self.__download_pretrained_model()
        self.__download_tf_record_creator()
        self.__generate_tf_records()



    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'Cars0.png')

    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

