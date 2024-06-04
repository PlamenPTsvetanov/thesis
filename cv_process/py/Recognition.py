from object_detection.builders import model_builder
from object_detection.utils import config_util
import tensorflow as tf
import os
import cv2
import numpy as np
from RecognitionSetup import Setup as meta


class Recognition:
    @tf.function
    def detect_fn(self, image):
        configs = config_util.get_configs_from_pipeline_file(meta.files['PIPELINE_CONFIG'])
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        #TODO preload model
        ckpt.restore(os.path.join(meta.paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    #TODO read path from env variable
    def scan_image(self, image_path):
        img = cv2.imread(image_path)
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # Махни го
        image_np_with_detections = image_np.copy()
        # Четене на номер
        # Запазване в базата
