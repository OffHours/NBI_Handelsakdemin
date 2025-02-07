#Imports
from random import randint
import cv2
import numpy as np
import sys
import joblib

def main():
    
    if True:
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("test")

        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("test", frame)

            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed+
                #cv2.imwrite('image.png', frame)
                img = frame
                break


        cam.release()

        cv2.destroyAllWindows()

    
    if img is None:
        sys.exit("Could not read the image.")

    
    # Converting the image to greyscale
    img_new = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_new = cv2.adaptiveThreshold(img_new, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 210, 14)

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
    
    cv2.imshow("Display window1", img)

    # Displaying the cropped image
    cv2.imshow("Display window", img_cropped)
    k = cv2.waitKey(0)
    
    #print(np.shape(img_small))
    
    # Saving the resized image as a png file when button S is pushed or closing applikation on Q
    if k == ord("s"):
        cv2.imwrite("dilated.png", img_small)
    elif k == ord("q"):
        cv2.destroyAllWindows()
    
    
    # Loading premade Classifier
    clf = joblib.load("et_clf_mnist_10000.pkl")
    
    # Reading scaled and filtered image
    img2 = cv2.imread("dilated.png", cv2.IMREAD_UNCHANGED)
    img2 = img2.reshape(1, -1)
    
    # Doing our prediction and printing it to the commandline
    proba = clf.predict_proba(img2)
    print(f'The number is: {clf.predict(img2)}. With a likelihood of: {(proba[0][clf.predict(img2)]*100)}%')

if __name__ == '__main__':
    main()