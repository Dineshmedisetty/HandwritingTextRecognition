import os
import urllib.request
import tarfile
import zipfile
import subprocess

def check_data_exists():
    """Check if required data files and directories already exist"""
    required_paths = [
        'data/words',  # Main data directory
        'data/words.txt',  # Words text file
        'IAM_Words'  # IAM Words directory
    ]
    return all(os.path.exists(path) for path in required_paths)

def download_and_extract_data():
    """Download and extract the IAM Words dataset"""
    print("Downloading and extracting IAM Words dataset...")
    
    # Download the dataset
    url = "https://github.com/sayakpaul/Handwriting-Recognizer-in-Keras/releases/download/v1.0.0/IAM_Words.zip"
    zip_path = "IAM_Words.zip"
    urllib.request.urlretrieve(url, zip_path)
    print("Dataset downloaded successfully.")

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    print("ZIP file extracted.")

    # Create data directory
    os.makedirs('data/words', exist_ok=True)

    # Extract words.tgz
    with tarfile.open('IAM_Words/words.tgz', 'r:gz') as tar:
        tar.extractall('data/words')
    print("Words archive extracted.")

    # Move words.txt to data directory
    import shutil
    shutil.move('IAM_Words/words.txt', 'data/')
    print("Words.txt moved to data directory.")

    # Clean up zip file
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("Cleaned up temporary files.")

# Check if data exists, if not download and extract
if not check_data_exists():
    download_and_extract_data()
else:
    print("IAM Words dataset already exists. Skipping download and extraction.")

# Preview the dataset (first 20 lines)
with open('data/words.txt', 'r') as f:
    for i, line in enumerate(f):
        if i < 20:
            print(line.strip())

from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

np.random.seed(42)
tf.random.set_seed(42)
base_path = "data"
words_list = []

words = open(f"{base_path}/words.txt", "r").readlines()
for line in words:
    if line[0]=='#':
        continue
    if line.split(" ")[1]!="err": # We don't need to deal with errored entries.
        words_list.append(line)

len(words_list)
np.random.shuffle(words_list)
split_idx = int(0.9 * len(words_list))
train_samples = words_list[:split_idx]
test_samples = words_list[split_idx:]

val_split_idx = int(0.5 * len(test_samples))
validation_samples = test_samples[:val_split_idx]
test_samples = test_samples[val_split_idx:]

assert len(words_list) == len(train_samples) + len(validation_samples) + len(test_samples)

print(f"Total training samples: {len(train_samples)}")
print(f"Total validation samples: {len(validation_samples)}")
print(f"Total test samples: {len(test_samples)}")

# We start building our data input pipeline by first preparing the image paths.
base_image_path = os.path.join(base_path, "words")

def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []
    for (i, file_line) in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path =  os.path.join(base_image_path, partI,
            partI + "-" + partII,
            image_name + ".png"
        )
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples


train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
test_img_paths, test_labels = get_image_paths_and_labels(test_samples)
# Find maximum length and the size of the vocabulary in the training data.
train_labels_cleaned = []
characters = set()
max_len = 0

for label in train_labels:
    label = label.split(" ")[-1].strip()
    for char in label:
        characters.add(char)

    max_len = max(max_len, len(label))
    train_labels_cleaned.append(label)

print("Maximum length: ", max_len)
print("Vocab size: ", len(characters))
# Check some label samples.
train_labels_cleaned[11:20]
def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels


validation_labels_cleaned = clean_labels(validation_labels)
test_labels_cleaned = clean_labels(test_labels)
from tensorflow.keras.layers import StringLookup
AUTOTUNE = tf.data.AUTOTUNE
# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)
def distortion_free_resize(image, img_size):
    w, h = img_size
    image  = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
                  [pad_height_top, pad_height_bottom],
                  [pad_width_left, pad_width_right],
                  [0, 0]
                ]
        )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

batch_size = 64
padding_token = 99
image_width = 128
image_height = 32


def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)

train_ds = prepare_dataset(train_img_paths, train_labels_cleaned)
validation_ds = prepare_dataset(validation_img_paths, validation_labels_cleaned)
test_ds = prepare_dataset(test_img_paths, test_labels_cleaned)
for data in train_ds.take(1):
    images, labels = data["image"], data["label"]

    _, ax = plt.subplots(4, 4, figsize=(15, 8))

    for i in range(16):
        img = images[i]
        img = tf.image.flip_left_right(img)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = (img * 255.).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]

        # Gather indices where label!= 99.
        label = labels[i]
        indices = tf.gather(label, tf.where(tf.math.not_equal(label, padding_token)))
        # Convert to string.
        label = tf.strings.reduce_join(num_to_char(indices))
        label = label.numpy().decode("utf-8")

        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")


plt.show()
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def build_model():
    # Inputs to the model
    input_img =  keras.Input(
        shape=(image_width, image_height, 1), name="image")
    labels =  keras.layers.Input(name="label", shape=(None,))

    # First conv block.
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x =  keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block.
    x =  keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x =  keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model.
    new_shape = ((image_width // 4), (image_height // 4) * 64)
    x =  keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x =  keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x =  keras.layers.Dropout(0.2)(x)

    # RNNs.
    x =  keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x =  keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer (the tokenizer is char-level)
    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    x =  keras.layers.Dense(len(char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model.
    model =  keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
    )
    # Optimizer.
    opt = keras.optimizers.Adam()
    # Compile the model and return.
    model.compile(optimizer=opt)
    return model


# Get the model.
model = build_model()
model.summary()
epochs = 50 # To get good results this should be at least 50.

# Train the model
model = build_model()
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs,
)
# Get the prediction model by extracting layers till the output layer.
prediction_model = keras.models.Model(
    inputs=model.input,  # Use model.input to get the input layer
    outputs=model.get_layer(name="dense2").output
)
prediction_model.summary()
# A utility function to decode the output of the network.
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


#  Let's check results on some test samples.
for batch in test_ds.take(1):
    batch_images = batch["image"]
    batch_true_labels = batch["label"]

    # Create the second input (input length for each image), reshaping it to 2D
    input_length = np.ones(batch_images.shape[0]) * batch_images.shape[2]
    input_length = np.expand_dims(input_length, axis=-1)  # Make input_length 2D

    # Get predictions, providing both inputs
    preds = prediction_model.predict([batch_images, input_length])  # Pass both inputs here
    pred_texts = decode_batch_predictions(preds)

    # Visualization code remains the same
    _, ax = plt.subplots(4, 4, figsize=(15, 8))
    for i in range(16):
        img = batch_images[i]
        img = tf.image.flip_left_right(img)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = (img * 255.).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]

        title = f"Prediction: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")

plt.show()
import editdistance
import numpy as np

# Function to calculate Character Error Rate (CER)
def calculate_cer(true_labels, pred_labels):
    cer_scores = []
    for true, pred in zip(true_labels, pred_labels):
        # Edit distance is used to compute CER
        cer_score = editdistance.eval(true, pred) / float(len(true))
        cer_scores.append(cer_score)
    return np.mean(cer_scores)

# Evaluate on the test set
true_labels = []
pred_labels = []

# Iterate over the test dataset
for batch in test_ds.take(1):
    batch_images = batch["image"]
    batch_true_labels = batch["label"]

    # Get predictions
    # Predict using both inputs (batch_images and input_length)
    preds = prediction_model.predict([batch_images, input_length])
    pred_texts = decode_batch_predictions(preds)

    # Get true labels (convert integer sequence to string)
    true_texts = []
    for label in batch_true_labels:
        label = tf.gather(label, tf.where(tf.math.not_equal(label, padding_token)))
        true_texts.append(tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8"))

    true_labels.extend(true_texts)
    pred_labels.extend(pred_texts)

# Calculate Character Error Rate (CER)
cer_score = calculate_cer(true_labels, pred_labels)
print(f"Character Error Rate (CER): {cer_score:.4f}")
from sklearn.metrics import classification_report

# Function to get the classification report
def get_classification_report(true_labels, pred_labels):
    # Flatten the true and predicted labels to calculate classification metrics
    all_true_labels = []
    all_pred_labels = []
    for true, pred in zip(true_labels, pred_labels):
        # Ensure both labels have the same length for character-wise comparison
        min_len = min(len(true), len(pred))
        all_true_labels.extend(list(true[:min_len]))
        all_pred_labels.extend(list(pred[:min_len]))

    # Classification report requires a list of individual characters
    return classification_report(all_true_labels, all_pred_labels, labels=list(characters), zero_division=1)

# Get the classification report
report = get_classification_report(true_labels, pred_labels)
print("Classification Report:\n", report)
