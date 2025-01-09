# TensorFlow Lite Age and Gender Detection Android App

An Android application that uses TensorFlow Lite models to detect age and gender from images using the device camera or gallery images.

## Features

- Age and gender detection from:
  - Device camera captures
  - Gallery image selection
- Multiple acceleration options:
  - CPU inference
  - GPU acceleration support
  - NNAPI support for compatible devices
- Real-time inference time reporting
- User-friendly interface with:
  - Camera capture button
  - Gallery selection button
  - Model initialization controls
  - Hardware acceleration toggles

## Technical Stack

- Language: Kotlin
- ML Framework: TensorFlow Lite
- Architecture Components:
  - Coroutines for asynchronous operations
  - View Binding for UI interactions
  - TFLite delegates (GPU, NNAPI) for acceleration

## Models
Refer to my [CNN Age Gender Detection](https://github.com/haturusinghe/cnn-age-gender) repo for model implementations

The application uses two TensorFlow Lite models:
- Age Estimation Model (`model_age_vN_nonq.tflite`)
  - Input size: 200x200 pixels
  - Output: Normalized age value
- Gender Classification Model (`model_gender_nonq.tflite`)
  - Input size: 128x128 pixels
  - Output: Binary classification [Male, Female]

## Requirements

- Android Studio Arctic Fox or newer
- Minimum SDK: Android 6.0 (API level 23)
- Android device with:
  - Camera
  - Storage access
  - (Optional) GPU/NNAPI support for acceleration

## Setup and Installation

1. Clone the repository
2. Open in Android Studio
3. Sync Gradle dependencies
4. Build and run on your device

## Usage Instructions

1. Launch the app
2. Initialize the models:
   - Select desired acceleration options (GPU/NNAPI if available)
   - Click "Initialize Models" button
3. After initialization, use either:
   - Camera button to take a photo
   - Gallery button to select an image
4. View results:
   - Predicted age
   - Predicted gender
   - Inference time for both models

## Permissions

- `CAMERA`: For capturing photos
- `READ_EXTERNAL_STORAGE`: For gallery image selection

## Project Structure

Key files:
- `MainActivity.kt`: Main UI and camera/gallery handling
- `AgeEstimationModel.kt`: Age prediction model wrapper
- `GenderClassificationModel.kt`: Gender prediction model wrapper

## License

This project is licensed under the MIT License.
