import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib

# Point to your model file for predictions, must be a .pkl package
model_file = joblib.load("et_clf_mnist_10000.pkl")

def process_image(uploaded_file, threshold_value, subtract_num):
    """
    Process the uploaded image:
      - Convert to grayscale.
      - Apply thresholding.
      - Find the largest contour and crop it.
      - Resize the cropped digit to 28x28 pixels.
      - Adjust the crop area to include extra space around the digit.
    Returns both the original image (for display) and the processed digit image.
    """

    # Read the uploaded image file into a NumPy array using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  
    if img is None:
        st.error("Error reading the image file.")
        return None, None

    # Converting the image to greyscale
    img_new = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_new = cv2.adaptiveThreshold(img_new, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshold_value, subtract_num)

    # Extracting the dimension of the image to get the center
    height, width = img_new.shape[:2]
    img_cent_width = width/2
    img_cent_height = height/2


    # Contour detection to find object in image

    contours, hierarchy = cv2.findContours(image=img_new, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    
    # Excluding the largest contour since it's the outside edge
    max_area = (height*width*0.9)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_area]  # Filter out the largest contour
    
    # Draw contours on the original image, uncomment imwrite function to check results
    img_new = cv2.drawContours(image=img_new, 
                                  contours=contours, 
                                  contourIdx=-1, 
                                  color=(0, 0, 255), 
                                  thickness=5, 
                                  lineType=cv2.LINE_AA)

    #cv2.imwrite('contours.png', img_new)
    edges = cv2.Canny(img_new,100,200)
    #cv2.imwrite('edges.png', edges)
    x_org,y_org,w_org,h_org = cv2.boundingRect(edges)

    img_copy_1 = img_new.copy()
    image_rect = cv2.rectangle(img_copy_1, (x_org, y_org), (x_org+w_org, y_org+h_org), color=(0, 255, 255))
    #cv2.imwrite('img_rect.png', image_rect)
    

    # Recalculating the Bounding Rectangle to get a square image
    h = int(h_org)
    w = h
    x = int(x_org-((w-w_org)/2))
    y = int(y_org)
    
    # Uncomment to the the coordinates and size of Bounding square 
    # print(f'first x y w h: {x,y,w,h}')
    img_copy_2 = img_new.copy()
    image_rect_2 = cv2.rectangle(img_copy_2, (x, y), (x+w, y+h), color=(0, 255, 255))
    #cv2.imwrite('img_square.png', image_rect_2)
    
    # Getting the center of the number
    num_cent_width = x+(w/2)
    num_cent_height = y+(h/2)

    # Shifting array to center the number before cropping

    shift_h = int(img_cent_width-num_cent_width) #The horizontal shift
    shift_v = int(img_cent_height-num_cent_height)  #The vertical shift


    # Uncomment imwrite functions to check if the code code the number properly
    image_rect = img_new.copy()
    edges = cv2.Canny(img_new,100,200)
    #cv2.imwrite('edges_2.png', edges)

    image_rect = cv2.rectangle(image_rect, (x, y), (x+w, y+w), color=(0, 255, 255))
    #cv2.imwrite('not_shifted.png', image_rect)

    img_new = np.roll(img_new, shift=(shift_h, shift_v), axis=(1, 1))
    #cv2.imwrite('shifted.png', img_new)
            
    # Contour detection to find object in image
    edges = cv2.Canny(img_new,100,200)
    x_new,y_new,w_new,h_new = cv2.boundingRect(edges)

    # Getting the center of the number
    num_cent_width_new = x_new+(w_new/2)
    num_cent_height_new = y_new+(h_new/2)

    # Shifting the Bounding box to follow the number
    x = int(x_new-((w/2)-(w_new/2)))
    y = int(y_new-((num_cent_height_new-num_cent_height)))
              
    # Cropping of the image, uncommnet imwrite to check result of crop
    img_copy_3 = img_new.copy()
    image_rect_3 = cv2.rectangle(img_copy_3, (x, y), (x+w, y+h), color=(0, 255, 255))
    # cv2.imwrite('img_square_cropped.png', image_rect_3)

    img_cropped = img_new[y:(y+h), x:(x+w)]
        
    img_cropped = cv2.copyMakeBorder(img_cropped, 75, 75, 75, 75, cv2.BORDER_CONSTANT, None, value=[255, 255, 255])
    #cv2.imwrite('img_cropped.png', img_cropped)

    # Resizing the cropped image to 28x28px to match mnist data.
    img_small = cv2.resize(img_cropped, (28, 28), interpolation = cv2.INTER_AREA)
    
    # Inverting and dilating image to match Mnist data
    img_small = cv2.bitwise_not(img_small)

    # Dilation, thickening of the lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_small = cv2.dilate(img_small, kernel, iterations=1)
    
    return img, img_small

def main():
    st.title("Digit Recognition App")
    
    # Sidebar for image processing options and file upload
    st.sidebar.header("Processing Options")
    threshold_value = st.sidebar.slider("Blocksize Value", 0, 255, 200)
    subtract_num = st.sidebar.slider("Subtract Constant", 1, 30, 14)
    if threshold_value%2 != 1:
        threshold_value +=1
    
    st.sidebar.header("File Uploads")
    image_file = st.sidebar.file_uploader("Upload a digit image", type=["jpg", "jpeg", "png"])
    
    
    if image_file is not None:
        # Process the image with the given threshold
        original_img, processed_digit = process_image(image_file, threshold_value, subtract_num)
        
        if original_img is not None:
            st.subheader("Original Image")
            st.image(original_img, caption="Uploaded Image", use_container_width =True)
            
        if processed_digit is not None:
            st.subheader("Processed Image (28×28 grayscale)")
            # Display the processed image. Specify channels="GRAY" so that Streamlit knows it is grayscale.
            st.image(processed_digit, caption="Processed Image", width=200, channels="GRAY")
            
            # If a model has been uploaded, make a prediction.
            if model_file is not None:
                try:
                    model = model_file
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    return
                
                # Preprocess the processed digit for prediction:
                # Flatten the 28x28 image into a 784-length vector and normalize pixel values.
                digit_flat = processed_digit.reshape(1, -1)
                                
                try:
                    prediction = model.predict(digit_flat)[0]
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    return
                
                st.subheader("Prediction")
                st.write(f"**Predicted Digit:** {prediction}")
                
                # If the model supports probability estimates, display the confidence percentage.
                try:
                    proba = model.predict_proba(digit_flat)
                    confidence = np.max(proba) * 100
                    st.write(f"**Confidence:** {confidence:.2f}%")
                except Exception:
                    st.info("Model does not support probability estimates.")
    else:
        st.info("Please upload an image file from the sidebar to begin.")

if __name__ == "__main__":
    main()
