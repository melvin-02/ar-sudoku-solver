import tensorflow as tf
import cv2
import numpy as np
import pickle
import math
import time
from sudoku_validator import isValidConfig
#from sudoku_solver import solve
from solver import solve_wrapper

#load the created model
model = tf.keras.models.load_model('models/digitOCR.h5')


def preprocess(img):
    '''
        This funciton perfroms basic image preprocessing to make it easy to find contours
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    adaptThresh_inv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    opening = cv2.morphologyEx(adaptThresh_inv, cv2.MORPH_OPEN, kernel)
    return opening


def find_largest_contour(image):
    '''
        The sudoku box will have the largest contour area. This funciton checks for the contour with the largest area and returns the 
        largest contour
    '''
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    biggest = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            if area > max_area:
                max_area = area
                biggest = contour
            #cv2.drawContours(original, [biggest], 0, (0,255,0), 1)
    return biggest
    

def get_corners(biggest_contour):
    '''
        This funciton returns the 4 corner coordinates of the grid.
        In opencv the origin starts at the top left corner of the image,
        so the x axis INCREASES to the RIGHT and DECREASES to the LEFT, 
            and y aixis INCREASES downwards and DECREASES upwards
        
        Hence, the sum of topleft coordinate will have least magnitude and  the sum of bottomright coordinate will have highest magnitude.
        Also the differnece of topright coordinate will have least magnitude and the difference of the bottomleft coordinate will have 
            the highest magnitude.

        We return a numpy list of coordinates in the order - [topleft, topright, bottomleft, bottomright] 
        THE ORDER IS IMPORTANT
    '''
    coords = np.zeros((4,2), np.float32)
    sumation = biggest_contour.sum(axis=2)
    coords[0] = biggest_contour[np.argmin(sumation)][0]     #topleft
    coords[2] = biggest_contour[np.argmax(sumation)][0]     #bottomright

    difference = np.diff(biggest_contour, axis=2)
    coords[1] = biggest_contour[np.argmin(difference)][0]   #topright
    coords[3] = biggest_contour[np.argmax(difference)][0]   #bottomleft

    return coords


def validate_rect(coords):
    '''
        This function checks if the 4 coordinates form a rectanlge (almost) or not.
            The sudoku grid will be a quadrilateral with almost equal opposite sides 

        The points will not form a perfect rectangle so we check if the length of oppposite sides are almost equal
            i.e. the length of smaller side is at least greater than 80% length of the larger side.
        
        If the condition fails then the points do not form a sudoku grid else the points form a sudoku grid.
    '''
    tleft, tright, bright, bleft = coords
    
    # using distance formula to calculate the width and height from the 4 coordinates
    widthTop = np.sqrt( ((tright[0] - tleft[0])**2) + ((tright[1] - tleft[1])**2) )
    widthBot = np.sqrt( ((bright[0] - bleft[0])**2) + ((bright[1] - bleft[1])**2) )

    heightRight = np.sqrt(((tright[0] - bright[0]) ** 2) + ((tright[1] - bright[1]) ** 2))
    heightLeft = np.sqrt(((tleft[0] - bleft[0]) ** 2) + ((tleft[1] - bleft[1]) ** 2))

    # the differnce between the lengths of opposited sides must be less than 20% (100-80) of the lenght of the larger side
    deltaH = 0.2 * max(heightLeft, heightRight)
    deltaW = 0.2 * max(widthBot, widthTop)

    if abs(widthTop-widthBot)<deltaW and abs(heightRight-heightLeft)<deltaH:
        return True
    return False


def perspective_transform(coords, image):
    '''
        This funtion returns a birds eye view of the extracted sudoku grid from the frame
    '''
    tleft, tright, bright, bleft = coords

    widthTop = np.sqrt( ((tright[0] - tleft[0])**2) + ((tright[1] - tleft[1])**2) )
    widthBot = np.sqrt( ((bright[0] - bleft[0])**2) + ((bright[1] - bleft[1])**2) )
    maxWidth = max(int(widthBot), int(widthTop))

    heightRight = np.sqrt(((tright[0] - bright[0]) ** 2) + ((tright[1] - bright[1]) ** 2))
    heightLeft = np.sqrt(((tleft[0] - bleft[0]) ** 2) + ((tleft[1] - bleft[1]) ** 2))
    maxHeight = max(int(heightRight), int(heightLeft))

    # create a destination array with points [topleft, topright, bottomright, bottomleft]
    # The topleft corner is the origin.
    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32" ) 

    M = cv2.getPerspectiveTransform(coords, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def remove_border(binary_image):
    '''
    This function removes the boundary pixels of the image
    '''
    x = binary_image.shape[1]
    y = binary_image.shape[0]
    border = int(0.12 * x)
    roi = binary_image[border:y-border, border:x-border]
    return roi


def empty(image):
    '''
        The digits are written in black on white background.
        If only less than 3% of the pixels are black, we can safely assume that the image is empty (only contains some noise),
            else the image contains a digit.
        The countNonZero function returns the number of non black pixels, that in our case is white pixels. So it is returning 
            the number of white pixels. If white pixels contain more than 97%(100-3) of the image, we declare it to be empty
    '''
    if cv2.countNonZero(image) >= 0.97*(image.shape[0] * image.shape[1]):
        return True
    else:
        return False


def extract_digit(grid):
    '''
        This function takes the sudoku grid, identifies the digits in the image and returns a numpy matrix of the predicted sudoku puzzle
    '''
    grid_resized = grid.copy()
    #resize image to a square so that we can divide it into 9x9 parts evenly.
    grid_resized = cv2.resize(grid_resized, (grid_resized.shape[0], grid_resized.shape[0]), cv2.INTER_AREA)
    posx = grid_resized.shape[1] // 9
    posy = grid_resized.shape[0] // 9
    border = 3
    digitSize = 32
    sudoku = np.zeros((9,9), dtype=np.uint8)

    #traverse through each part of the 9x9 puzzle and predict the number in that region
    for i in range(9):
        for j in range(9):
            # extract the digit at the particular location 
            digit = grid_resized[posy*i : posy*(i+1), posx*j : posx*(j+1)]

            # to check if the block is empty or conatins a digit, extract the center of the image and perform the empty function on it.
            #   if the block contains a digit the center of the iamge will have black pixels
            #   if the block is blank then the center of the image will be mostly white pixels.
            thresholdY = int(0.25 * digit.shape[1])
            thresholdX = int(0.25 * digit.shape[0])
            center = digit[thresholdY: digit.shape[1]-thresholdY, thresholdX: digit.shape[0]-thresholdX]
            if empty(center):
                # if the block is empty skip (do nothing)
                continue
            else:
                # if block contains digit, remove border pixels
                crop_image = remove_border(digit)
                #reisize the image to the input size of prediction model - few border pixels
                resize = cv2.resize(crop_image, (digitSize-2*border, digitSize-2*border), cv2.INTER_AREA)
                #we pad the image with white border as the images used in the model training have some white border pixels
                padded_digit = cv2.copyMakeBorder(resize, border, border, border, border, cv2.BORDER_CONSTANT, value=(255,255,255))
                padded_digit = padded_digit.astype('float32')
                padded_digit = padded_digit/255.0
                # the model contains 9 classes which start from 0 to 8. The digits in sudoku however range from 1-9.
                # So we add 1 to the prediciton to get the correct number.
                pred = model.predict(padded_digit.reshape(1,digitSize,digitSize,1)).argmax(axis=1)[0] + 1
                # store the predicted value at its index position
                sudoku[i][j] = pred
                
    return sudoku


def fill_sudoku(solved, unsolved, img, debug=False):
    '''
        This funciton is used to fill the warped sudoku image with the solution.
            The funcion expects a solved sudoku matrix, an unsolved sudoku matrix, and the warped image
            The debug parameter can be used to draw the predicted numbers as well. This can be useful to see what the model is
                actually predicting
    '''
    # First we calculate the width and height of the warped image.
    gridw = img.shape[1]
    gridh = img.shape[0]

    #Divide the width and height by 9 to get the block locations
    xgap = gridw // 9
    ygap = gridh // 9
    # added a small margin value to fit the text values a littel more better in their respective blocks
    margin = int(0.015 * img.shape[1])

    for i in range(9):
        for j in range(9):
            #only write those numbers which are solved
            if unsolved[i][j] == 0:         
                text = str(solved[i][j])
                xloc = xgap*j + margin
                yloc = ygap*(i+1) - margin
                fontsize = gridw / 400
                cv2.putText(img, text, (xloc, yloc), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255, 0,0), 2)
            
            # if debug is ON, also print the numbers which are predicted
            elif debug :
                text = str(solved[i][j])
                xloc = xgap*j + margin
                yloc = ygap*(i+1) - margin
                fontsize = gridw / 400
                cv2.putText(img, text, (xloc, yloc), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0,2552,0), 2)
            
    return img

def unwarp_image(img_src, img_dest, pts_dest):
    '''
        This function is used to warp the solution image onto the actual frame.
    '''
    pts_dest = np.array(pts_dest)

    height, width = img_src.shape[0], img_src.shape[1]
    pts_source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(pts_source, pts_dest)
    warped = cv2.warpPerspective(img_src, h, (img_dest.shape[1], img_dest.shape[0]))
    cv2.fillConvexPoly(img_dest, pts_dest.astype('int32'), 0)

    dst_img = cv2.add(img_dest, warped)

    return dst_img

def main():
    '''
        This is where the whole procedure takes place.
        The steps are:
            -> preprocess the frame 
            -> find largest contour (which is expected to be the sudoku grid box) 
            -> find the corners of the largest contour 
            -> check if the corners approximately form a rectangle 
            -> extract the grid image 
            -> divide it into 9x9 blocks and perfrom digit prediction to find sudoku matrix 
            -> check if resultant sudoku matrix is valid  
            -> solve the sudoku 
            -> write the result on the extracted warped image 
            -> place this final solved image onto the frame 
            -> show the frame.
    '''
    #initialize an empty 9x9 matrix
    sudoku_matrix = np.zeros((9,9), dtype=np.uint8)
    #set a boolean flag to false
    validation = False
    
    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        processedFrame = preprocess(frame)
        biggest = find_largest_contour(processedFrame)
        try:
            coords = get_corners(biggest)
            if validate_rect(coords):   
                #for i in range(4):
                #    cv2.circle(frame, (int(coords[i][0]), int(coords[i][1])), 5, (0,0,255), -1)  
                cv2.drawContours(frame, [biggest], 0, (0,255,0), 2)
                
                warped = perspective_transform(coords, frame)
                warped_binary = preprocess(warped)
                warped_inv = cv2.bitwise_not(warped_binary)
                if not validation:
                    sudoku_matrix = extract_digit(warped_inv)
                    unsolved = sudoku_matrix.copy()
                    if isValidConfig(sudoku_matrix) and np.count_nonzero(sudoku_matrix)!=0:
                        validation = True
                        sudoku_matrix, solve_time = solve_wrapper(sudoku_matrix)
                solved_grid_image = fill_sudoku(sudoku_matrix, unsolved, warped)
                frame = unwarp_image(solved_grid_image, frame, coords)
        
        except :
            pass
                
        # used to calculate the fps
        fps = int ( 1/ (time.time() -start_time) ) 
        fps = str(fps) 
        cv2.putText(frame, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('live', frame)
    
        #exit if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()        
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()