"""
Image classifier with Grad-CAM.

Part 1 of the Image Classification project.
  1. Loads MobileNetV2, a network already trained on the ImageNet dataset.
  2. Classifies an image and prints the top-3 predictions with confidence.
  3. Runs Grad-CAM to build a heatmap of the regions the model relied on,
     and saves that heatmap blended over the original image.

Run:  python3 base_classifier.py
Then type an image filename such as  myphoto.jpg
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Hide TensorFlow's info and warning messages so the output is easy to read
tf.get_logger().setLevel("ERROR")

# MobileNetV2 with ImageNet weights: knows 1,000 everyday object classes
model = MobileNetV2(weights="imagenet")

# The last convolutional layer of MobileNetV2. It still has a 7x7 spatial
# grid, so it can tell us WHERE in the image the evidence came from.
LAST_CONV_LAYER = "Conv_1"

# A second model that shares the same weights but returns two things:
# the feature maps of the last conv layer, and the final predictions.
grad_model = Model(
    inputs=model.inputs,
    outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output],
)


def load_and_prepare(image_path):
    """Load an image and turn it into the 1x224x224x3 array MobileNetV2 expects."""
    img = image.load_img(image_path, target_size=(224, 224))  # resize to 224x224
    img_array = image.img_to_array(img)                       # pixels -> numbers
    img_array = preprocess_input(img_array)                   # scale to -1..1
    return np.expand_dims(img_array, axis=0)                  # add batch dimension


def make_gradcam_heatmap(img_array, class_index=None):
    """Return a 7x7 Grad-CAM heatmap (values 0..1) for one image.

    Grad-CAM steps:
      1. Run the image through the network and keep the last conv feature maps.
      2. Take the gradient of the chosen class score with respect to those maps.
         A large gradient means "changing this feature map changes the score a lot".
      3. Average each map's gradient into one weight per feature map.
      4. Add up the feature maps, each multiplied by its weight.
      5. Keep only the positive part (ReLU) and scale to 0..1.
    """
    with tf.GradientTape() as tape:
        conv_maps, preds = grad_model(tf.convert_to_tensor(img_array))
        if isinstance(preds, (list, tuple)):   # some Keras versions wrap the output in a list
            preds = preds[0]
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))   # default: the top predicted class
        class_score = preds[:, class_index]

    grads = tape.gradient(class_score, conv_maps)          # step 2
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))        # step 3: one weight per map
    heatmap = tf.reduce_sum(conv_maps[0] * weights, axis=-1)  # step 4
    heatmap = tf.maximum(heatmap, 0)                       # step 5: ReLU

    max_value = tf.reduce_max(heatmap)
    if max_value > 0:
        heatmap = heatmap / max_value                      # scale to 0..1
    return heatmap.numpy()


def save_gradcam_overlay(image_path, heatmap, output_path, alpha=0.4):
    """Blend the heatmap over the original image and save it."""
    original = cv2.imread(image_path)                      # original size, BGR colours
    if original is None:
        raise ValueError(f"OpenCV could not read '{image_path}'")

    # Stretch the 7x7 heatmap to the size of the photo
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    # JET colour map: blue = ignored, red = most important
    coloured = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Weighted blend: (1 - alpha) of the photo + alpha of the heatmap
    overlay = cv2.addWeighted(original, 1 - alpha, coloured, alpha, 0)
    cv2.imwrite(output_path, overlay)


def classify_image(image_path):
    """Classify one image, print the top-3 predictions and save a Grad-CAM overlay."""
    try:
        img_array = load_and_prepare(image_path)

        predictions = model.predict(img_array, verbose=0)
        top3 = decode_predictions(predictions, top=3)[0]

        print("\nTop-3 Predictions for", image_path)
        for i, (_, label, score) in enumerate(top3):
            print(f"  {i + 1}: {label} ({score:.2f})")

        # Explain the number 1 prediction with Grad-CAM
        heatmap = make_gradcam_heatmap(img_array)
        output_path = image_path.rsplit(".", 1)[0] + "_gradcam.jpg"
        save_gradcam_overlay(image_path, heatmap, output_path)
        print(f"Grad-CAM overlay for '{top3[0][1]}' saved as '{output_path}'.")
    except Exception as e:
        print(f"Error processing '{image_path}': {e}")


if __name__ == "__main__":
    print("Image Classifier with Grad-CAM (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename: ").strip()
        if image_path.lower() == "exit":
            print("Goodbye!")
            break
        classify_image(image_path)
