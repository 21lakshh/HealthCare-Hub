import cv2
import imutils
import pickle
import os
import tensorflow as tf
from PIL import Image
import numpy as np


# loading all the saved models 

# loading the bone fracture model
working_dir = os.path.dirname(os.path.abspath(__file__)) 
model_path = f"{working_dir}/bonefracture_model.h5"
bonefracture_model = tf.keras.models.load_model(model_path)

# loading the diabetes model 
diabetes_model = pickle.load(open(f'{working_dir}/diabetes_trained_model.sav','rb'))
diabetes_scalar = pickle.load(open(f'{working_dir}/diabetes_scaler.sav','rb'))
breastcancer_scaler = pickle.load(open(f'{working_dir}/breastcancer_scaler.sav','rb'))

# loading the heart disease model 
heart_disease_model = pickle.load(open(f'{working_dir}/heart_disease_model.sav','rb'))

# loading the calories burnt model 
calories_burnt_model = pickle.load(open(f'{working_dir}/caloriesburnt_training_model.sav','rb'))

# loading the asthma model 
asthma_model = pickle.load(open(f'{working_dir}/asthma_model.sav','rb'))

# loading the medical insurance model 
medicalinsurance_model = pickle.load(open(f'{working_dir}/medicalinsurance_trained_model.sav','rb'))

# loading the breast cancer model 
model_path = f'{working_dir}/breastcancer_model.h5'
breastcancer_model = tf.keras.models.load_model(model_path)


brain_tumor_model_path = f'{working_dir}/braintumour_detection.h5'
brain_tumor_model = tf.keras.models.load_model(brain_tumor_model_path)

def crop_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # covnerting it into a black and white image
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0) # blurring to remove noise and smoothen image for detecting contours
    img_thresh = cv2.threshold(img_blur, 45, 255, cv2.THRESH_BINARY)[1] # pixels below 45 become black and above 255 become white, used for segmentation so they stand out from background
    # converts it into a bianry format black or white
    img_thresh = cv2.erode(img_thresh, None, iterations=2) # removes small noise and shrinks white regions
    img_thresh = cv2.dilate(img_thresh, None, iterations=2) # expands white regions

    contours = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # contours are outlines of binary image
    contours = imutils.grab_contours(contours)
    # cv2.RETR_EXTERNAL: Retrieves only the outermost contours
    # cv2.CHAIN_APPROX_NONE: Stores all contour points without compression.
    # imutils.grab_contours(contours): Ensures compatibility with different OpenCV versions.

    if not contours:
        return image  # Return original image if no contour is found

    c = max(contours, key=cv2.contourArea) # finding extreme points of the contours
    extLeft = tuple(c[c[:, :, 0].argmin()])[0]
    extRight = tuple(c[c[:, :, 0].argmax()])[0]
    extTop = tuple(c[c[:, :, 1].argmin()])[0]
    extBottom = tuple(c[c[:, :, 1].argmax()])[0]

    new_img = image[extTop[1]: extBottom[1], extLeft[0]:extRight[0]] # cropping with extreme points
    return new_img


def preprocess_image_for_brain_tumor_detection(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure it's an RGB image
    input_data_as_numpy_array = np.asarray(image, dtype=np.uint8)  # Convert to uint8
    cropped_img = crop_image(input_data_as_numpy_array)  # Using your crop_image function
    resized_img = cv2.resize(cropped_img, (240, 240))
    return resized_img


def make_prediction_for_brain_tumor_detection(processed_image):
  img = np.expand_dims(processed_image,axis=0)
  prediction = brain_tumor_model.predict(img)
  predicted_class_index = np.argmax(prediction)
  class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Replace with your actual labels
  predicted_class_name = class_labels[predicted_class_index]

  return predicted_class_name
    
def make_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = diabetes_scalar.transform(input_data_reshaped)
    prediction = diabetes_model.predict(std_data)

    if prediction[0] == 1:
        return 'Person is Diabetic!'
    else:
        return 'Person is not Diabetic!'
    


def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array