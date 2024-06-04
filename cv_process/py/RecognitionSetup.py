import wget
import os
import subprocess
import tarfile
import sys
from zipfile import ZipFile
import shutil


class Setup:
    CUSTOM_MODEL_NAME = 'custom_ssd_mobilenet_v2'
    PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'

    paths = {
        'WORKSPACE_PATH': os.path.join('..', 'workspace'),
        'SCRIPTS_PATH': os.path.join('..', 'scripts'),
        'APIMODEL_PATH': os.path.join('..', 'models'),
        'ANNOTATION_PATH': os.path.join('..', 'workspace', 'annotations'),
        'IMAGE_PATH': os.path.join('..', 'workspace', 'images'),
        'MODEL_PATH': os.path.join('..', 'workspace', 'models'),
        'PRETRAINED_MODEL_PATH': os.path.join('..', 'workspace', 'pre-trained-model'),
        'CHECKPOINT_PATH': os.path.join('..', 'workspace', 'models', CUSTOM_MODEL_NAME),
        'OUTPUT_PATH': os.path.join('..', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
        'TFJS_PATH': os.path.join('..', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
        'TFLITE_PATH': os.path.join('..', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
        'PROTOC_PATH': os.path.join('..', 'protoc')
    }

    files = {
        'PIPELINE_CONFIG': os.path.join('..', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
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
                print(os.getcwd())
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
        if not os.listdir(os.path.join(self.paths['APIMODEL_PATH'], 'research', 'object_detection')):
            subprocess.run(['git', 'clone', 'https://github.com/tensorflow/models', 'models'], check=True)

    def __download_pretrained_model(self):
        if not os.listdir(self.paths['PRETRAINED_MODEL_PATH']):
            wget.download(self.PRETRAINED_MODEL_URL)
            os.replace(os.path.join(os.getcwd(), self.PRETRAINED_MODEL_NAME + '.tar.gz'),
                       os.path.join(self.paths['PRETRAINED_MODEL_PATH'], self.PRETRAINED_MODEL_NAME + '.tar.gz'))

            with tarfile.open(
                    os.path.join(self.paths['PRETRAINED_MODEL_PATH'], self.PRETRAINED_MODEL_NAME + '.tar.gz')) as file:
                file.extractall(path=os.path.join(self.paths['PRETRAINED_MODEL_PATH']))

    def __download_tf_record_creator(self):
        if not os.listdir(self.paths['SCRIPTS_PATH']):
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

        shutil.copy(os.path.join(self.paths['PRETRAINED_MODEL_PATH'], self.PRETRAINED_MODEL_NAME,
                                 'pipeline.config'), os.path.join(self.paths['CHECKPOINT_PATH']))


    def setup(self):
        print("Creating paths...")
        self.__create_paths()
        print("Downloading protoc...")
        self.__download_protoc()
        print("Downloading object detection...")
        self.__download_object_detection()
        print("Downloading pretrained model...")
        self.__download_pretrained_model()
        print("Downloading tf record creator...")
        self.__download_tf_record_creator()
        print("Generating tf records...")
        self.__generate_tf_records()
        print("Setup is done...")
