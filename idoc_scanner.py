import numpy as np
import cv2
import sys

imageHeight = 1100
imageWidth = imageHeight * 800 // 1100 # MIMIC A4 PAPER DIMENSION

def resize_for_display(img, max_height=600):
    height, width = img.shape[:2]
    if height > max_height:
        scale = max_height / height
        new_width = int(width * scale)
        return cv2.resize(img, (new_width, max_height))
    return img

def main():
    if len(sys.argv) != 3:
        print("Usage: python idoc_scanner.py <input image> <output image>")
        sys.exit(0)
    
    black = resize_for_display(np.zeros((imageHeight, imageWidth, 3), np.uint8))

    # IMAGE IMPORT
    image = cv2.imread(sys.argv[1])
    if image is None:
        print(f"Error: Could not find image {sys.argv[1]}")
        sys.exit(1)

    # IMAGE PREPROCESSING
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 55) # VERY BLUR TO SHOW ONLY DOCUMENT EDGE
    edges = cv2.Canny(blurred, 150, 255)
    kernel = np.ones((5,5))
    dila =cv2.dilate(edges, kernel, iterations=2)
    edges =cv2.erode(dila, kernel, iterations=1)
    
    # DRAW CONTOUR IN IMAGE
    image_all_contour = image.copy()
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_all_contour, contours, -1, (0, 255, 0), 10)

    # FIND LARGEST CONTOUR AND VERTICES
    max_area = 0
    biggest = None
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            temp = cv2.approxPolyDP(i, 0.02 * cv2.arcLength(i, True), True)
            if area > max_area and len(temp) == 4:
                biggest = temp
                max_area = area
    if biggest is None:
        print("No document found! Using entire image as fallback.")
        h, w = image.shape[:2]
        # Create corners for entire image
        biggest = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.int32)
        warped = cv2.resize(gray, (imageWidth, imageHeight))

    # REORDER VERTICES FOR WARP
    biggest = biggest.reshape((4,2))
    temp = np.zeros((4,1,2), dtype=np.int32)
    
    diff = np.diff(biggest, axis=1)
    temp[1] = biggest[np.argmin(diff)]
    temp[2] = biggest[np.argmax(diff)]

    add = biggest.sum(1)
    temp[0] = biggest[np.argmin(add)]
    temp[3] = biggest[np.argmax(add)]

    biggest = temp

    # WARPING
    if biggest.size != 0:
        points1 = np.float32(biggest)
        points2 = np.float32([[0,0],[imageWidth,0],[0,imageHeight],[imageWidth,imageHeight]])
        matrix = cv2.getPerspectiveTransform(points1, points2)
        warped = cv2.warpPerspective(gray, matrix,(imageWidth, imageHeight))
    # HERE WARPED USES GRAY FOR POSTPROCESSING, NORMALLY DIRECT USE IMAGE

    # IMAGE POSTPROCESSING
    result = warped.copy()
    """
    result = cv2.adaptiveThreshold(result, 255, 1, 1, 3, 2)
    result = cv2.bitwise_not(result)
    """
    I_sharpen = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
    result = cv2.filter2D(result, -1, I_sharpen)

    # DISPLAY IMAGE FOR DEBUGGING
    """
    gray_small = resize_for_display(gray)
    gray_small = cv2.cvtColor(gray_small, cv2.COLOR_GRAY2BGR)
    blurred_small = resize_for_display(blurred)
    blurred_small = cv2.cvtColor(blurred_small, cv2.COLOR_GRAY2BGR)
    warped_small = resize_for_display(warped)
    warped_small = cv2.cvtColor(warped_small, cv2.COLOR_GRAY2BGR)
    
    edges_small = resize_for_display(edges)
    edges_small = cv2.cvtColor(edges_small, cv2.COLOR_GRAY2BGR)
    original_small = resize_for_display(image)
    image_all_contour_small = resize_for_display(image_all_contour)
    final_small = resize_for_display(result)
    final_small = cv2.cvtColor(final_small, cv2.COLOR_GRAY2BGR)
    top_left = np.hstack((original_small, edges_small))
    top_right = np.hstack((image_all_contour_small, final_small))
    combined = np.hstack((top_left, top_right))

    cv2.imshow('Intelligent Document Scanner', combined)
    """
    
    # SAVE restored.jpg
    if result is not None:
        cv2.imwrite(sys.argv[2], result)
        print("Image restored.jpg generated successfully.\n")
    else:
        print("Error: No output image found.\n")

    # SAVE combined Image for comparison
    
    if result is not None:
        image_resized = cv2.resize(image, (imageWidth, imageHeight))
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        imageCombined = np.hstack((image_resized, result))
        
        cv2.imwrite("imageCombined.jpg", imageCombined)
        print("Image imageCombined.jpg generated successfully.\n")
    else:
        print("Error: Could not create combined image.\n")

if __name__ == "__main__":
    main()