import tensorflow_datasets as tfds
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# Load the ImageNet-R dataset
builder = tfds.builder('imagenet_r')
builder.download_and_prepare()
imagenet_r = builder.as_dataset(split='test')

# Load the pre-trained model
model = tf.keras.applications.ResNet50(weights='imagenet')


def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    return decoded_predictions


images_dir = os.path.join(os.path.dirname(__file__), 'images')

# Classify each image in the directory
for img_file in os.listdir(images_dir):
    img_path = os.path.join(images_dir, img_file)
    if os.path.isfile(img_path):
        predictions = classify_image(img_path)
        print(f"Predictions for {img_file}:")
        for pred in predictions:
            print(f"{pred[1]}: {pred[2]*100:.2f}%")
        print()
