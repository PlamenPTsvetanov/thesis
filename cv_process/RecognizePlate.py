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
import easyocr as eo
import pytesseract as pyt
import keras_ocr
from google.protobuf import text_format
from PIL import Image
from matplotlib import pyplot as plt




CUSTOM_MODEL_NAME = 'custom_ssd_mobilenet_v2'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('workspace'),
    'SCRIPTS_PATH': os.path.join('scripts'),
    'APIMODEL_PATH': os.path.join('models'),
    'ANNOTATION_PATH': os.path.join('workspace', 'annotations'),
    'IMAGE_PATH': os.path.join('workspace', 'images'),
    'MODEL_PATH': os.path.join('workspace', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('workspace', 'pre-trained-model'),
    'CHECKPOINT_PATH': os.path.join('workspace', 'models', CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH': os.path.join('workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH': os.path.join('workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH': os.path.join('protoc')
}

files = {
    'PIPELINE_CONFIG': os.path.join('workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

labels = [{'name': 'licence', 'id': 1}]


def create_paths():
    for path in paths.values():
        if not os.path.exists(path):
            os.mkdir(path)


def download_protoc():
    if not os.listdir(paths['PROTOC_PATH']):
        print('Downloading protoc...')
        url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.19.5/protoc-3.19.5-win64.zip"
        wget.download(url)
        os.replace(os.path.join(os.getcwd(), "protoc-3.19.5-win64.zip"),
                   os.path.join(paths['PROTOC_PATH'], "protoc-3.19.5-win64.zip"))
        with ZipFile(os.path.join(paths['PROTOC_PATH'], "protoc-3.19.5-win64.zip"), 'r') as protoc:
            protoc.extractall(paths['PROTOC_PATH'])
        try:
            os.chdir('models/research')
            subprocess.run(["protoc", "object_detection/protos/*.proto", "--python_out=."], check=True, shell=True)
            subprocess.run(["copy", "object_detection\\packages\\tf2\\setup.py", "setup.py"], check=True, shell=True)
            subprocess.run([sys.executable, "setup.py", "build"], check=True)
            subprocess.run([sys.executable, "setup.py", "install"], check=True)
            os.chdir('slim')
            subprocess.run(["pip", "install", "-e", "."], check=True, shell=True)
            os.chdir(os.path.join("..", "..", ".."))
        except subprocess.CalledProcessError as e:
            print("An exception occurred!" + str(e))

        os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))
        print(os.environ['PATH'])


def download_object_detection():
    if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
        subprocess.run(['git', 'clone', 'https://github.com/tensorflow/models', 'models'], check=True)


def download_pretrained_model():
    if not os.path.exists(paths['PRETRAINED_MODEL_PATH']):
        os.mkdir(paths['PRETRAINED_MODEL_PATH'])
        wget.download(PRETRAINED_MODEL_URL)
        os.replace(os.path.join(os.getcwd(), PRETRAINED_MODEL_NAME + '.tar.gz'),
                   os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'))

        with tarfile.open(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz')) as file:
            file.extractall(path=os.path.join(paths['PRETRAINED_MODEL_PATH']))


def download_tf_record_creator():
    if not os.path.exists(files['TF_RECORD_SCRIPT']):
        subprocess.run(['git', 'clone', 'https://github.com/nicknochnack/GenerateTFRecord', paths['SCRIPTS_PATH']],
                       check=True)


def generate_tf_records():
    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')
    os.system(sys.executable + " " + files['TF_RECORD_SCRIPT']
              + ' -x ' + os.path.join(paths['IMAGE_PATH'], 'test')
              + ' -l ' + files['LABELMAP']
              + ' -o ' + os.path.join(paths['ANNOTATION_PATH'], 'test.record'))
    os.system(sys.executable + " " + files['TF_RECORD_SCRIPT']
              + ' -x ' + os.path.join(paths['IMAGE_PATH'], 'train')
              + ' -l ' + files['LABELMAP']
              + ' -o ' + os.path.join(paths['ANNOTATION_PATH'], 'train.record'))
    os.system("copy "
              + os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')
              + " "
              + os.path.join(paths['CHECKPOINT_PATH']))


def setup():
    create_paths()
    # download_protoc()
    # download_object_detection()
    # download_pretrained_model()
    # download_tf_record_creator()
    # generate_tf_records()

def config_pipeline():
    global config, pipeline_config
    config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as file:
        proto_str = file.read()
        text_format.Merge(proto_str, pipeline_config)
    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'],
                                                                     PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
        os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
        os.path.join(paths['ANNOTATION_PATH'], 'test.record')]
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
        f.write(config_text)


@tf.function
def detect_fn(image):
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


setup()

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

config_pipeline()

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

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.7,
            agnostic_mode=False)

plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.show()