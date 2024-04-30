from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

loaded_model = load_model('E:\\Uni\\Term 7 - Fall 1402\\Courses\\Robotics\\Project\\mavic_main\\CNN_model')


def predict_image(image:np.array, modell):
  image = image.reshape(1, 28, 28, 1)
  classes = ['T-Shirts', 'Pants', 'Pullovers', 'Shoes & Sandals', 'Bags']
  return classes[np.argmax(modell.predict(image, verbose=0))]


def predict_image_by_path(path:str):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

def get_label(path):
    # Load the image
    image = cv2.imread(path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the lower and upper thresholds for gray intensity
    lower_gray = 1  # Adjust this value based on your requirement
    upper_gray = 53  # Adjust this value based on your requirement

    # Create a binary mask for gray regions
    mask = cv2.inRange(gray, lower_gray, upper_gray)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the minimum bounding rectangle that encloses the contour
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))

    # Make the width and height equal to the maximum of the two dimensions
    max_dim = max(w, h)
    w = max_dim
    h = max_dim

    # Calculate the new top-left corner to make the bounding box square
    x = x + int((w - max_dim) / 2)
    y = y + int((h - max_dim) / 2)

    # Draw the contour rectangle on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract the region of interest (ROI) from the original image based on the modified bounding box
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cropped_image = cv2.resize(roi, (28, 28))
    # print(cropped_image.shape)
    # plt.imshow(cropped_image, cmap='gray')
    # plt.show()

    # print(predict_image(cropped_image, loaded_model))

    # Save the cropped image with the square contour
    # cv2.imwrite('output5.jpg', roi)

    # Plot the image with the contour rectangle
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Bounding Box')
    plt.axis('off')
    plt.show()
    return predict_image(cropped_image, loaded_model)
    
if __name__ == "__main__":   
    image_path = 'E:\\Uni\\Term 7 - Fall 1402\\Courses\\Robotics\\Project\\mavic_main\\webots\\mavic\\controllers\\mavic2pro_pyth'
    for i in range(5):
        image_name = f'image{i}.jpg'
        path = image_path + image_name
        get_label(path)
