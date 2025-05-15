# Handwriting Recognition Web Application

This is a web application that uses deep learning to recognize handwritten text from images. The application is built with TensorFlow and Flask, featuring a modern UI for easy interaction.

## Features

- Upload images via drag-and-drop or file selection
- Real-time image preview
- Modern, responsive UI using Tailwind CSS
- Server-side processing with TensorFlow
- Error handling and user feedback

## Setup
1. Create and activate a virtual environment
``` bash
python -m venv venv
venv\Scripts\activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train_model.py
```
This will create a `model` directory containing the trained model and character mappings.

4. Start the web application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Click the upload area or drag and drop an image containing handwritten text
2. Preview the image
3. Click "Recognize Text" to process the image
4. View the recognized text result

## Model Architecture

The model uses a CNN-RNN architecture with CTC loss for handwriting recognition:
- Convolutional layers for feature extraction
- Bidirectional LSTM layers for sequence processing
- CTC layer for text alignment and prediction

## Notes

- The model is trained on the IAM Words dataset
- For best results, use clear images of handwritten text
- Note: The model was trained for 50 epochs to achieve optimal accuracy and minimize character error rate.
- Supported image formats: PNG, JPG, GIF
- Maximum file size: 10MB

## Results

The model achieved the following results on the test set:
- Accuracy: 88%
- Character Error Rate (CER): 0.09182

These results demonstrate strong performance in recognizing handwritten text, even in cases with moderate noise or style variation. The low CER indicates that the system is effective at minimizing transcription mistakes at the character level.

## License

This project is open source and available under the MIT License. 
