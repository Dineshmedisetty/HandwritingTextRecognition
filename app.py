import os
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import StringLookup
import numpy as np
from PIL import Image
import io
import json
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)
CORS(app)  # Enable CORS for all routes

# Enable debug mode for better error messages
app.config['DEBUG'] = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Constants
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
IMG_SIZE = (128, 32)
MIN_IMAGE_SIZE = (32, 8)  # Minimum image dimensions

# Load the saved model and character mappings
model = None
char_to_num = None
num_to_char = None
prediction_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_and_mappings():
    global model, char_to_num, num_to_char, prediction_model
    try:
        # Define custom objects for model loading
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
                return y_pred

        # Load the model with custom objects
        model_path = os.path.join('model', 'handwriting_model.keras')
        logger.info(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects={'CTCLayer': CTCLayer}, compile=False)
        
        # Create prediction model
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input,
            model.get_layer(name="dense2").output
        )
        
        # Load character mappings
        mappings_path = os.path.join('model', 'char_mappings.json')
        logger.info(f"Loading mappings from: {mappings_path}")
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
        
        # Create StringLookup layers
        char_to_num = StringLookup(vocabulary=mappings['char_to_num'], mask_token=None)
        num_to_char = StringLookup(vocabulary=mappings['num_to_char'], mask_token=None, invert=True)
        
        logger.info("Model and mappings loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model and mappings: {str(e)}")
        raise

def calculate_confidence(pred_probs):
    """Calculate confidence score based on prediction probabilities"""
    try:
        # Get the maximum probability for each timestep
        max_probs = np.max(pred_probs, axis=-1)
        # Average probability across all timesteps
        confidence = float(np.mean(max_probs))
        return min(confidence * 100, 100.0)  # Convert to percentage and cap at 100
    except Exception as e:
        logger.error(f"Error calculating confidence: {str(e)}")
        return 0.0

def preprocess_image(image_data, img_size=IMG_SIZE):
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data)).convert('L')
        logger.debug(f"Original image size: {image.size}")
        
        # Basic image validation
        if image.size[0] < MIN_IMAGE_SIZE[0] or image.size[1] < MIN_IMAGE_SIZE[1]:
            raise ValueError(f"Image is too small. Minimum size is {MIN_IMAGE_SIZE[0]}x{MIN_IMAGE_SIZE[1]} pixels")
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Check for empty or corrupted images
        if image_array.size == 0 or np.all(image_array == 0) or np.all(image_array == 255):
            raise ValueError("Invalid image content: empty or corrupted image")
        
        # Convert to tensor
        image = tf.convert_to_tensor(image_array)
        if len(image.shape) == 2:
            image = tf.expand_dims(image, axis=-1)
        logger.debug(f"Image tensor shape after expansion: {image.shape}")
        
        # Resize and normalize
        image = tf.image.resize(image, size=img_size[::-1], preserve_aspect_ratio=True)
        logger.debug(f"Image shape after resize: {image.shape}")
        
        # Add padding
        pad_height = img_size[1] - tf.shape(image)[0]
        pad_width = img_size[0] - tf.shape(image)[1]
        
        if pad_height % 2 != 0:
            pad_height_top = pad_height // 2 + 1
            pad_height_bottom = pad_height // 2
        else:
            pad_height_top = pad_height_bottom = pad_height // 2
            
        if pad_width % 2 != 0:
            pad_width_left = pad_width // 2 + 1
            pad_width_right = pad_width // 2
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
        logger.debug(f"Image shape after padding: {image.shape}")
        
        # Ensure correct orientation and normalization
        image = tf.transpose(image, perm=[1, 0, 2])
        image = tf.image.flip_left_right(image)
        image = tf.cast(image, tf.float32) / 255.0
        
        # Check for NaN or Inf values
        if tf.reduce_any(tf.math.is_nan(image)) or tf.reduce_any(tf.math.is_inf(image)):
            raise ValueError("Invalid pixel values detected (NaN or Inf)")
        
        final_image = tf.expand_dims(image, axis=0)
        logger.debug(f"Final image shape: {final_image.shape}")
        return final_image
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        raise

def decode_batch_predictions(pred):
    try:
        logger.debug(f"Prediction shape: {pred.shape}")
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        
        # Use greedy search with proper input length
        results = keras.backend.ctc_decode(
            pred,
            input_length=input_len,
            greedy=True
        )[0][0]
        
        logger.debug(f"Decoded results shape: {results.shape}")
        
        output_text = []
        for res in results:
            # Filter out -1 padding tokens
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            # Convert to text
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            # Remove any special characters that might have been introduced
            res = ''.join(c for c in res if c.isprintable())
            output_text.append(res)
            
        logger.debug(f"Decoded text: {output_text}")
        return output_text
    except Exception as e:
        logger.error(f"Error in prediction decoding: {str(e)}")
        raise

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return f"Error loading page: {str(e)}", 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return "Internal Server Error: " + str(error), 500

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Validate file selection
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG, GIF'}), 400
    
    # Validate file size
    file_content = file.read()
    if len(file_content) > MAX_IMAGE_SIZE:
        return jsonify({'error': 'File size too large. Maximum size: 10MB'}), 400
    
    try:
        # Process the image
        logger.info("Processing uploaded image")
        processed_image = preprocess_image(file_content)
        
        # Get prediction
        logger.info("Running prediction")
        preds = prediction_model.predict(processed_image, verbose=0)
        logger.debug(f"Raw prediction shape: {preds.shape}")
        
        # Get the predicted text
        predicted_text = decode_batch_predictions(preds)[0]
        
        # Calculate confidence score
        confidence = calculate_confidence(preds[0])
        
        logger.info(f"Prediction result: {predicted_text} (confidence: {confidence:.2f}%)")
        
        # Validate prediction
        if not predicted_text or len(predicted_text.strip()) == 0:
            return jsonify({
                'error': 'No text detected in the image',
                'prediction': '',
                'confidence': 0.0
            }), 200
        
        return jsonify({
            'prediction': predicted_text,
            'confidence': confidence
        })
    
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the image'}), 500

if __name__ == '__main__':
    # Load model and mappings
    load_model_and_mappings()
    app.run(debug=True, host='0.0.0.0', port=5000) 