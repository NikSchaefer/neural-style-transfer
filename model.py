import os
import tensorflow as tf

os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow_hub as hub

mpl.rcParams["figure.figsize"] = (12, 12)
mpl.rcParams["axes.grid"] = False

import numpy as np
import PIL.Image


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    max_dim = 1200
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


CONTENT_IMG_PATH = "images/mars.png"
STYLE_IMG_PATH = "images/art.jpg"


def main():
    content_img = load_img(CONTENT_IMG_PATH)
    style_img = load_img(STYLE_IMG_PATH)

    plt.subplot(1, 2, 1)
    imshow(content_img, "Content Image")

    plt.subplot(1, 2, 2)
    imshow(style_img, "Style Image")
    # plt.show()

    hub_model = hub.load(
        "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    )
    print("Model loaded.")

    stylized_image = hub_model(tf.constant(image), tf.constant(style_img))
    img = tensor_to_image(stylized_image)

    plt.imshow(img)
    plt.show()
    # img.save("save/wave.png")


if __name__ == "__main__":
    main()
