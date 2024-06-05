import os
import re

import cv2
import numpy as np
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from RecognitionSetup import Setup as meta


class Recognition:

    def scan_image(self, image_path):
        img = cv2.imread(image_path)
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        print(os.getcwd())
        configs = config_util.get_configs_from_pipeline_file(meta.files['PIPELINE_CONFIG'])
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(meta.paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

        image, shapes = detection_model.preprocess(input_tensor)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        category_index = label_map_util.create_category_index_from_labelmap(meta.files['LABELMAP'])

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.7,
            agnostic_mode=False)

        # plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        # plt.show()

        return image_np_with_detections, detections

    @staticmethod
    def __clear_text(text):
        text = ''.join(text)
        text = re.sub(r'[^a-zA-Z0-9]', '', text)
        return text

    @staticmethod
    def __common_ocr(image, detections, detection_threshold, ocr_function):
        scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
        boxes = detections['detection_boxes'][:len(scores)]

        width = image.shape[1]
        height = image.shape[0]

        for idx, box in enumerate(boxes):
            roi = box * [height, width, height, width]
            region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]

            text = ocr_function(region)
            return text, region

    def ocr(self, image_np_with_detections, detections, detection_threshold, ocr_function):
        image = image_np_with_detections
        text, region = self.__common_ocr(image, detections, detection_threshold, ocr_function)
        if type(text) == tuple:
            text = text[0]
        return self.__clear_text(text), region
