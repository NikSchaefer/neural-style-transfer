# Generative Style Transfer

Implementation of Neural Style Transfer using a GAN(Generative Adversial Network) using the technique outlined in [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576).

## What is Neural Style Transfer

Neural Style Transfer is a technique that takes two images, a content image and a style image, and blends them together so the content image is in the style of the style image.

## Running

set the path to the content image and style image in the file.

```py
# path to images
CONTENT_IMG_PATH = "images/content/planets.jpg"
STYLE_IMG_PATH = "images/style/art.jpg"

# image will save every epoch under
# save/IMG_SAVE_NAME-{n}.png
IMG_SAVE_NAME = "mountain-greatwave"
```

You can alter the weights in the style_weight, content_weight, and total_variation_weight

```py
style_weight = 1e-2  # default: 1e-2
content_weight = 1e4  # default: 1e4
total_variation_weight = 30  # default: 30
```

## Examples

Running for around 10 epochs(100 steps) tends to yield the best results. With certain images it may be better to run more or less epochs as well as editing the weights to generate a better image.

The following example was run with 10 epochs with default weights.

![Blended Neural Style Image](./save/mountain-scream/mountain-art2-9.png)

The Content Image and Style Image for this example

Content Image:
![Content Image](./images/content/mountain.jpg)

Style Image:
![Style Image](./images/style/scream.jpg)

### Other Examples

Deepspace Scence style with La Muse Painting
![Deepspace-Art Painting](./save/deepspace-art/deepspace-art-9.png)

Sunset mountain style with Greatwave Painting
![Sunset Mountain Painting](./save/sunset-greatwave/sunset-greatwave-8.png)

## Dependencies

Install dependencies with Pip

`pip install tensorflow numpy Pillow`

Dependencies:

- Tensorflow
- Numpy
- Pillow

## License

[MIT](https://choosealicense.com/licenses/mit/)