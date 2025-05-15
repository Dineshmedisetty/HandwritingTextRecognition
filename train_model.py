import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import json

# Import the model architecture and training code from hwr.py
from hwr import (
    build_model,
    prepare_dataset,
    get_image_paths_and_labels,
    clean_labels,
    char_to_num,
    num_to_char,
    train_samples,
    validation_samples,
    test_samples,
)

def main():
    try:
        # Create model directory if it doesn't exist
        model_dir = 'model'
        os.makedirs(model_dir, exist_ok=True)
        
        # Get image paths and labels
        print("Loading datasets...")
        train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
        validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
        
        # Validate dataset
        if len(train_img_paths) == 0 or len(validation_img_paths) == 0:
            raise ValueError("Empty dataset detected. Please check the data loading process.")
        
        print(f"Training samples: {len(train_img_paths)}")
        print(f"Validation samples: {len(validation_img_paths)}")
        
        # Clean labels
        train_labels_cleaned = clean_labels(train_labels)
        validation_labels_cleaned = clean_labels(validation_labels)
        
        # Prepare datasets
        train_ds = prepare_dataset(train_img_paths, train_labels_cleaned)
        validation_ds = prepare_dataset(validation_img_paths, validation_labels_cleaned)
        
        # Build model
        model = build_model()
        
        # Configure optimizer with learning rate
        initial_learning_rate = 0.001
        optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
        model.compile(optimizer=optimizer)
        
        # Define callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when training plateaus
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=0.00001,
                verbose=1
            ),
            # Save the best model during training
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(model_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        # Train for fewer epochs during testing. Increase this for better results.
        epochs = 50  # Increased from 10 to 50 for better results
        
        print("Starting model training...")
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Create and save the prediction model
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input,
            model.get_layer(name="dense2").output
        )
        
        # Save the prediction model in Keras format
        model_path = os.path.join(model_dir, 'handwriting_model.keras')
        prediction_model.save(model_path, save_format='keras_v3')
        
        # Save character mappings
        char_to_num_vocab = char_to_num.get_vocabulary()
        num_to_char_vocab = num_to_char.get_vocabulary()
        
        # Save vocabularies as JSON
        mappings = {
            'char_to_num': list(char_to_num_vocab),
            'num_to_char': list(num_to_char_vocab)
        }
        
        mappings_path = os.path.join(model_dir, 'char_mappings.json')
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        
        # Save training history
        history_path = os.path.join(model_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            history_dict = {key: [float(x) for x in values] for key, values in history.history.items()}
            json.dump(history_dict, f, indent=2)
        
        # Save model summary
        summary_path = os.path.join(model_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {os.path.abspath(model_path)}")
        print(f"Character mappings saved to: {os.path.abspath(mappings_path)}")
        print(f"Training history saved to: {os.path.abspath(history_path)}")
        print(f"Model summary saved to: {os.path.abspath(summary_path)}")
        
        # Print final metrics
        final_val_loss = min(history.history['val_loss'])
        print(f"\nBest validation loss: {final_val_loss:.4f}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 