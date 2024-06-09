import os
import time

import tensorflow as tf
from matplotlib import pyplot as plt


class GenerativeEngine:
    paths = {
        'JOINED_DATASET_PATH': os.path.join(os.getcwd(), 'workspace', 'images', 'archive', 'dataset', 'blurred'),
        'WORKING_DATASET_PATH': os.path.join(os.getcwd(), 'workspace', 'images', 'archive', 'dataset', 'input'),
        'TEST_DATASET_PATH': os.path.join(os.getcwd(), 'workspace', 'images', 'archive', 'dataset', 'test'),
        'VALIDATE_DATASET_PATH': os.path.join(os.getcwd(), 'workspace', 'images', 'archive', 'dataset', 'validate'),
        'CHECKPOINT_PATH': os.path.join(os.getcwd(), 'workspace', 'models', 'pix2pix'),
        'LOG_PATH': os.path.join(os.getcwd(), 'workspace', 'logs'),
    }

    BUFFER_SIZE = 64
    BATCH_SIZE = 1
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    LAMBDA = 100

    @staticmethod
    def limit_gpu():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    @staticmethod
    def load_image(file_path):
        image = tf.io.read_file(file_path)
        image = tf.io.decode_jpeg(image)

        w = tf.shape(image)[1]
        w = w // 2
        real_image = image[:, w:, :]
        input_image = image[:, :w, :]

        real_image = tf.image.resize(real_image, [256, 256])
        input_image = tf.image.resize(input_image, [256, 256])

        real_image = tf.cast(real_image, tf.float32)
        input_image = tf.cast(input_image, tf.float32)

        return input_image, real_image

    @staticmethod
    def resize(input_image, real_image, height, width):
        input_image = tf.image.resize(input_image, [height, width],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image

    def random_crop(self, input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, self.IMG_HEIGHT, self.IMG_WIDTH, 3])

        return cropped_image[0], cropped_image[1]

    @staticmethod
    def normalize(input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image

    def random_jitter(self, input_image, real_image):
        input_image, real_image = self.resize(input_image, real_image, 286, 286)

        input_image, real_image = self.random_crop(input_image, real_image)
        input_image, real_image = self.random_crop(input_image, real_image)

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def load_image_train(self, image_file):
        input_image, real_image = self.load_image(image_file)
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def load_image_test(self, image_file):
        input_image, real_image = self.load_image(image_file)
        input_image, real_image = self.resize(input_image, real_image,
                                              self.IMG_HEIGHT, self.IMG_WIDTH)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def load_train_dataset(self):
        train_dataset = tf.data.Dataset.list_files(os.path.join(self.paths['WORKING_DATASET_PATH'], '*.jpg'))
        train_dataset = train_dataset.map(self.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.BUFFER_SIZE)
        train_dataset = train_dataset.batch(self.BATCH_SIZE)

        return train_dataset

    def load_test_dataset(self):
        test_dataset = tf.data.Dataset.list_files(os.path.join(self.paths['TEST_DATASET_PATH'], '*.jpg'))
        test_dataset = test_dataset.map(self.load_image_test)
        test_dataset = test_dataset.batch(self.BATCH_SIZE)

        return test_dataset

    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer,
                                   use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    @staticmethod
    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    def Generator(self):
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])

        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),
            self.downsample(128, 4),
            self.downsample(256, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4),
            self.upsample(256, 4),
            self.upsample(128, 4),
            self.upsample(64, 4),
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(3, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh')

        x = inputs

        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    @staticmethod
    def loss_object():
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def generator_loss(self, disc_generated_output, gen_output, target):
        # Adversarial loss
        loss = self.loss_object()
        gan_loss = loss(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean Squared Error (MSE) loss
        mse_loss = tf.reduce_mean(tf.square(target - gen_output))

        # Total generator loss
        total_gen_loss = gan_loss + (self.LAMBDA * 0.1 * mse_loss)

        return total_gen_loss, gan_loss, mse_loss

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])

        down1 = self.downsample(64, 4, False)(x)
        down2 = self.downsample(128, 4)(down1)
        down3 = self.downsample(256, 4)(down2)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        loss = self.loss_object()
        real_loss = loss(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = loss(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss


    def checkpoint(self, generator, discriminator):
        checkpoint_prefix = os.path.join(self.paths['CHECKPOINT_PATH'], "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                                         discriminator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                                         generator=generator,
                                         discriminator=discriminator)
        return checkpoint, checkpoint_prefix

    def train_step(self, input_image, target, step, generator, discriminator):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    def fit(self, train_ds, test_ds, steps):
        example_input, example_target = next(iter(test_ds.take(1)))
        start = time.time()
        generator = self.Generator()
        discriminator = self.Discriminator()

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if (step) % 1000 == 0:
                if step != 0:
                    print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')
                start = time.time()

                self.generate_images(generator, example_input, example_target)
                print(f"Step: {step // 1000}k")

            self.train_step(input_image, target, step, generator, discriminator)
            # Training step
            if (step + 1) % 10 == 0:
                print('.', end='', flush=True)

            # Save (checkpoint) the model every 5k steps
            if (step + 1) % 5000 == 0:
                checkpoint, chk_index = self.checkpoint(generator, discriminator)
                checkpoint.save(file_prefix=chk_index)

    def generate_images(self, model, test_input, tar):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()

    @staticmethod
    def run():
        engine = GenerativeEngine()
        train_dataset = engine.load_train_dataset()
        test_dataset = engine.load_test_dataset()
        engine.fit(train_dataset, test_dataset, steps=10000)


GenerativeEngine.run()
