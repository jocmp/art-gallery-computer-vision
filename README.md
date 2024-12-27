# Art Gallery Computer Vision

A proof of concept using TensorFlow and the [imagenet_r](https://www.tensorflow.org/datasets/catalog/imagenet_r) dataset to classify images from the GVSU Art Gallery collection.

The ImageNet-R dataset is "a set of images labelled... by collecting art, cartoons, deviantart, graffiti, embroidery, graphics, origami, paintings, patterns, plastic objects, plush objects, sculptures, sketches, tattoos, toys, and video games". This makes it a good candidate dataset to demonstrate how image classification can work.

Below is the output of 5 images from the collection stored in the [gallery/images](./gallery/images) directory. As you can see there are some classifications that work very well like "lakeside: 77.10%" with the _Gulls of Leland_ painting. Others are less accurate classifications like the "oxcart" in _Picnic at Macatawa_.

```
make install
make run
```

**Output**

```
Predictions for autumn.jpeg:
valley: 50.67%
lakeside: 34.42%
castle: 3.70%

Predictions for camelia.jpeg:
overskirt: 31.56%
hoopskirt: 18.11%
groom: 12.17%

Predictions for gulls.jpg:
lakeside: 77.10%
breakwater: 12.34%
seashore: 3.71%

Predictions for picnic.jpg:
oxcart: 7.41%
cliff: 6.85%
wreck: 6.72%

Predictions for sunset.jpeg:
tray: 62.12%
ant: 5.22%
geyser: 4.66%
```
