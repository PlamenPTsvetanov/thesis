import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import restoration


class RLDeblur:

    @staticmethod
    def richardson_lucy_deconv(noisy_channel, psf, num_iter=20):
        return restoration.richardson_lucy(noisy_channel, psf[:, :, 0, 0], num_iter=num_iter)

    def deblur(self, path, output_path, format):
        image = np.array(Image.open(path)).astype(np.float32)

        image /= 255.0

        psf = tf.constant(np.ones((3, 3)) / 25, dtype=tf.float32)
        psf = tf.reshape(psf, [3, 3, 1, 1])

        # Prepare the PSF for convolution
        psf = psf.numpy().squeeze()
        psf = tf.expand_dims(psf, 2)
        psf = tf.expand_dims(psf, 3)
        psf = psf.numpy()

        deconvolved_rl = np.zeros_like(image)
        for i in range(3):
            deconvolved_rl[:, :, i] = self.richardson_lucy_deconv(image[:, :, i], psf)

        deconvolved_uint8 = (deconvolved_rl[:, :, :3] * 255).astype(np.uint8)

        # Apply denoising
        denoised = cv2.fastNlMeansDenoisingColored(deconvolved_uint8, None, 10, 10, 7, 21)

        img = Image.fromarray(deconvolved_uint8)

        new_path = os.path.join(output_path, "_deblur." + format)

        img.save(new_path)
        return new_path
