import tensorflow as tf
import cv2
import numpy as np
import pickle
import math
import time
from sudoku_validator import isValidConfig
from sudoku_solver import solve
from solver import solve_wrapper

model = tf.keras.models.load_model('models/digitOCR.h5')

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    adaptThresh_inv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    opening = cv2.morphologyEx(adaptThresh_inv, cv2.MORPH_OPEN, kernel)
    return opening


def find_largest_contour(image):
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

    coords = np.zeros((4,2), np.float32)
    sumation = biggest_contour.sum(axis=2)
    coords[0] = biggest_contour[np.argmin(sumation)][0]     #topleft
    coords[2] = biggest_contour[np.argmax(sumation)][0]     #bottomright

    difference = np.diff(biggest_contour, axis=2)
    coords[1] = biggest_contour[np.argmin(difference)][0]   #topright
    coords[3] = biggest_contour[np.argmax(difference)][0]   #bottomleft

    return coords


def validate_rect(coords):
    tleft, tright, bright, bleft = coords
    
    widthTop = np.sqrt( ((tright[0] - tleft[0])**2) + ((tright[1] - tleft[1])**2) )
    widthBot = np.sqrt( ((bright[0] - bleft[0])**2) + ((bright[1] - bleft[1])**2) )

    heightRight = np.sqrt(((tright[0] - bright[0]) ** 2) + ((tright[1] - bright[1]) ** 2))
    heightLeft = np.sqrt(((tleft[0] - bleft[0]) ** 2) + ((tleft[1] - bleft[1]) ** 2))

    deltaH = 0.2 * max(heightLeft, heightRight)
    deltaW = 0.2 * max(widthBot, widthTop)

    if abs(widthTop-widthBot)<deltaW and abs(heightRight-heightLeft)<deltaH:
        return True
    return False


def perspective_transform(coords, image):

    tleft, tright, bright, bleft = coords

    widthTop = np.sqrt( ((tright[0] - tleft[0])**2) + ((tright[1] - tleft[1])**2) )
    widthBot = np.sqrt( ((bright[0] - bleft[0])**2) + ((bright[1] - bleft[1])**2) )
    maxWidth = max(int(widthBot), int(widthTop))

    heightRight = np.sqrt(((tright[0] - bright[0]) ** 2) + ((tright[1] - bright[1]) ** 2))
    heightLeft = np.sqrt(((tleft[0] - bleft[0]) ** 2) + ((tleft[1] - bleft[1]) ** 2))
    maxHeight = max(int(heightRight), int(heightLeft))

    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32" ) 

    M = cv2.getPerspectiveTransform(coords, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def unwarp_image(img_src, img_dest, pts_dest):
    pts_dest = np.array(pts_dest)

    height, width = img_src.shape[0], img_src.shape[1]
    pts_source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(pts_source, pts_dest)
    warped = cv2.warpPerspective(img_src, h, (img_dest.shape[1], img_dest.shape[0]))
    cv2.fillConvexPoly(img_dest, pts_dest.astype('int32'), 0)

    dst_img = cv2.add(img_dest, warped)

    return dst_img


def remove_border(binary_image):
    x = binary_image.shape[1]
    y = binary_image.shape[0]
    border = int(0.12 * x)
    roi = binary_image[border:y-border, border:x-border]
    return roi


def empty(image):
    
    if cv2.countNonZero(image) >= 0.97*(image.shape[0] * image.shape[1]):
        return True
    else:
        return False


def extract_digit(grid):
    grid_resized = grid.copy()
    grid_resized = cv2.resize(grid_resized, (grid_resized.shape[0], grid_resized.shape[0]), cv2.INTER_AREA)
    posx = grid_resized.shape[1] // 9
    posy = grid_resized.shape[0] // 9
    border = 3
    digitSize = 32
    sudoku = np.zeros((9,9), dtype=np.uint8)

    for i in range(9):
        for j in range(9):
            digit = grid_resized[posy*i : posy*(i+1), posx*j : posx*(j+1)]
            #digit = largest_connected_component(digit)
            #digit = remove_border(digit)
            thresholdY = int(0.25 * digit.shape[1])
            thresholdX = int(0.25 * digit.shape[0])
            center = digit[thresholdY: digit.shape[1]-thresholdY, thresholdX: digit.shape[0]-thresholdX]
            if empty(center):
                continue
            else:
                crop_image = remove_border(digit)
                '''
                ratio = 0.4     
                # Top
                while np.sum(crop_image[0]) <= (1-ratio) * crop_image.shape[1] * 255:
                    crop_image = crop_image[1:]
                # Bottom
                while np.sum(crop_image[:,-1]) <= (1-ratio) * crop_image.shape[1] * 255:
                    crop_image = np.delete(crop_image, -1, 1)
                # Left
                while np.sum(crop_image[:,0]) <= (1-ratio) * crop_image.shape[0] * 255:
                    crop_image = np.delete(crop_image, 0, 1)
                # Right
                while np.sum(crop_image[-1]) <= (1-ratio) * crop_image.shape[0] * 255:
                    crop_image = crop_image[:-1]
                '''
                resize = cv2.resize(crop_image, (digitSize-2*border, digitSize-2*border), cv2.INTER_AREA)
                padded_digit = cv2.copyMakeBorder(resize, border, border, border, border, cv2.BORDER_CONSTANT, value=(255,255,255))
                padded_digit = padded_digit.astype('float32')
                padded_digit = padded_digit/255.0
                pred = model.predict(padded_digit.reshape(1,digitSize,digitSize,1)).argmax(axis=1)[0] + 1
                sudoku[i][j] = pred
                
    return sudoku


def fill_sudoku(solved, unsolved, img, debug=False):

    tleft, tright, bright, bleft = [[0,0], [img.shape[1]-1, 0], [img.shape[1]-1, img.shape[0]-1], [0, img.shape[0]-1]]
    gridw = int(max(abs(tright[0]-tleft[0]),abs(bright[0] - bleft[0])))
    gridh = int(max(abs(tleft[1]- bleft[1]),abs(tright[1] - bright[1])))

    xgap = gridw // 9
    ygap = gridh // 9
    margin = int(0.015 * img.shape[1])

    for i in range(9):
        for j in range(9):
            if unsolved[i][j] == 0:         
                text = str(solved[i][j])
                xloc =  int(tleft[0]) + xgap*j + margin
                yloc = int(tleft[1]) + ygap*(i+1) - margin
                fontsize = gridw / 400
                cv2.putText(img, text, (xloc, yloc), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0,255,0), 2)
            
            else :
                text = str(solved[i][j])
                xloc =  int(tleft[0]) + xgap*j + margin
                yloc = int(tleft[1]) + ygap*(i+1) - margin
                fontsize = gridw / 400
                cv2.putText(img, text, (xloc, yloc), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255,0,0), 2)
            
    return img



def main():

    sudoku_matrix = np.zeros((9,9), dtype=np.uint8)
    validation = False
    prev_sudoku = None
    prev_frame_time = 0
    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        ret, frame = cap.read()
        processedFrame = preprocess(frame)
        biggest = find_largest_contour(processedFrame)
        try:
            coords = get_corners(biggest)
            if validate_rect(coords):   
                #for i in range(4):
                #    cv2.circle(frame, (int(coords[i][0]), int(coords[i][1])), 5, (0,0,255), -1)  
                #cv2.drawContours(frame, [biggest], 0, (0,255,0), 1)
                
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
        
        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        fps = int(fps) 
        fps = str(fps) 
        cv2.putText(frame, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('live', frame)
    

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()        
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()