# Prediction function for all images in a folder
def predict_characters_in_folder(model, folder_path, image_size):
    predictions = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            try:
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (image_size, image_size))
                img = img.reshape(1, image_size, image_size, 1) / 255.0
                prediction = model.predict(img)
                predicted_class = categories[np.argmax(prediction)]
                predictions[img_path] = predicted_class
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    return predictions

# Path to the folder containing the images to be predicted
input_folder = "/content/drive/My Drive/ML Project/output"  # Change this to your input folder path

# Predict characters in the input folder
predictions = predict_characters_in_folder(model, input_folder, image_size)

# Print the predictions
for img_path, predicted_class in predictions.items():
    print(f"Image: {img_path} --> Predicted Character: {predicted_class}")
