import os
import sys

import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.utils import config_util

from RecognitionSetup import Setup as meta


class Training:

    @staticmethod
    def __config_pipeline():
        config_util.get_configs_from_pipeline_file(meta.files['PIPELINE_CONFIG'])
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(meta.files['PIPELINE_CONFIG'], "r") as file:
            proto_str = file.read()
            text_format.Merge(proto_str, pipeline_config)

        pipeline_config.model.ssd.num_classes = len(meta.labels)
        pipeline_config.train_config.batch_size = 4
        pipeline_config.train_config.fine_tune_checkpoint = os.path.join(meta.paths['PRETRAINED_MODEL_PATH'],
                                                                         meta.PRETRAINED_MODEL_NAME, 'checkpoint',
                                                                         'ckpt-0')
        pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
        pipeline_config.train_input_reader.label_map_path = meta.files['LABELMAP']
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
            os.path.join(meta.paths['ANNOTATION_PATH'], 'train.record')]
        pipeline_config.eval_input_reader[0].label_map_path = meta.files['LABELMAP']
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
            os.path.join(meta.paths['ANNOTATION_PATH'], 'test.record')]

        config_text = text_format.MessageToString(pipeline_config)
        with tf.io.gfile.GFile(meta.files['PIPELINE_CONFIG'], "wb") as f:
            f.write(config_text)

    def train_model(self):
        self.__config_pipeline()
        training_script = os.path.join(meta.paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
        command = ("{} {} --model_dir={} --pipeline_config_path={} --num_train_steps=10000"
                   .format(sys.executable,
                           training_script,
                           meta.paths['CHECKPOINT_PATH'],
                           meta.files['PIPELINE_CONFIG']))
        os.system(command)

