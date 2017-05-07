import cv2
import os
import numpy as np
from line import Line
import math

camera_cal_dir = './camera_cal'
camera_cal_dim = (9, 6)
# Define the region
area_of_interest = [[150 + 430, 460], [1150 - 440, 460], [1150, 720], [150, 720]]
left_lane = Line()
right_lane = Line()


def imread(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)


def calibrate(images_dir, dimension=(9, 6)):
    cal_img_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
    cal_imgs = [cv2.imread(path) for path in cal_img_paths]
    print('Found {} calibration images'.format(len(cal_imgs)))
    gray_cal_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in cal_imgs]

    object_points = []
    image_points = []

    # image (w, h, color)
    objp = np.zeros((dimension[0] * dimension[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dimension[0], 0:dimension[1]].T.reshape(-1, 2)

    # finding image points on the calibration images
    for idx, img in enumerate(gray_cal_imgs):
        ret, corners = cv2.findChessboardCorners(img, dimension, None)
        print("Processed {} image".format(idx))
        if ret:
            image_points.append(corners)
            object_points.append(objp)

    print("Finished processing all chessboard corners")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        gray_cal_imgs[0].shape[::-1],
        None, None)

    print("Calibration finished")
    return ret, mtx, dist, rvecs, tvecs


# undistort image
def undistort(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)


# Perspective transform
def warper(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image


def abs_gradient(gray, sobel_kernel=3):
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    return grad_x, grad_y


def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Take the absolute value of the derivative or gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take the gradient in x and y separately
    grad_x, grad_y = abs_gradient(gray, sobel_kernel=sobel_kernel)
    # Calculate the magnitude
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(mag) / 255
    mag = (mag / scale_factor).astype(np.uint8)
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(mag)
    binary_output[(mag >= mag_thresh[0]) & (mag <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate the x and y gradients
    grad_x, grad_y = abs_gradient(image, sobel_kernel=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    abs_grad_dir = np.arctan2(np.absolute(grad_y), np.absolute(grad_x))
    binary_output = np.zeros_like(abs_grad_dir)
    binary_output[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def corners_unwarp(img, mtx, dist):
    undistorted = undistort(img, mtx, dist)
    # Choose an offset from image corners to plot detected corners
    offset1 = 200  # offset for dst points x value
    offset2 = 0  # offset for dst points bottom y value
    offset3 = 0  # offset for dst points top y value

    img_size = (img.shape[1], img.shape[0])
    # For source points I'm grabbing the outer four detected corners
    src = np.float32(area_of_interest)

    dst = np.float32([[offset1, offset3],
                      [img_size[0] - offset1, offset3],
                      [img_size[0] - offset1, img_size[1] - offset2],
                      [offset1, img_size[1] - offset2]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undistorted, M, img_size)
    # Return the resulting image and matrix
    return warped, M, Minv


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def pipeline(img):
    # Gaussian Blur
    kernel_size = 5
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]
    # Gray scale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Define sobel kernel size
    ksize = 7
    # Apply each of the threshold functions
    grad_x = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(10, 255))
    grad_y = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(60, 255))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(40, 255))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(.65, 1.05))
    # Combine all the threshold information
    combined = np.zeros_like(dir_binary)
    combined[((grad_x == 1) & (grad_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # Threshold color channel
    s_binary = np.zeros_like(combined)
    s_binary[(s > 160) & (s < 255)] = 1
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.zeros_like(combined)
    color_binary[(s_binary > 0) | (combined > 0)] = 1
    # Defining vertices for marked area
    imshape = img.shape
    left_bottom = (100, imshape[0])
    right_bottom = (imshape[1] - 20, imshape[0])
    apex1 = (610, 410)
    apex2 = (680, 410)
    inner_left_bottom = (310, imshape[0])
    inner_right_bottom = (1150, imshape[0])
    inner_apex1 = (700, 480)
    inner_apex2 = (650, 480)
    vertices = np.array([[left_bottom, apex1, apex2,
                          right_bottom, inner_right_bottom,
                          inner_apex1, inner_apex2, inner_left_bottom]], dtype=np.int32)
    # Masked area
    color_binary = region_of_interest(color_binary, vertices)
    return color_binary


def find_curvature(y_vals, fit_x):
    # Define y-value where we want radius of curvature
    # Choosing the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(y_vals) / 2
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    fit_cr = np.polyfit(y_vals * ym_per_pix, fit_x * xm_per_pix, 2)
    curverad = ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    return curverad


def find_position(pts, shape):
    # Find the position of the car from the center
    # It will show if the car is 'x' meters from the left or right
    position = shape[1] / 2
    left = np.min(pts[(pts[:, 1] < position) & (pts[:, 0] > 700)][:, 1])
    right = np.max(pts[(pts[:, 1] > position) & (pts[:, 0] > 700)][:, 1])
    center = (left + right) / 2
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    return (position - center) * xm_per_pix


def find_nearest(array, value):
    # Function to find the nearest point from array
    if len(array) > 0:
        idx = (np.abs(array - value)).argmin()
        return array[idx]


def find_peaks(image, y_window_top, y_window_bottom, x_left, x_right):
    # Find the historgram from the image inside the window
    histogram = np.sum(image[y_window_top:y_window_bottom, :], axis=0)
    # Find the max from the histogram
    if len(histogram[int(x_left):int(x_right)]) > 0:
        return np.argmax(histogram[int(x_left):int(x_right)]) + x_left
    else:
        return (x_left + x_right) / 2


def sanity_check(lane, curved, fitx, fit):
    # Sanity check for the lane
    if lane.detected:  # If lane is detected
        # If sanity check passes
        if abs(curved / lane.radius_of_curvature - 1) < .6:
            lane.detected = True
            lane.current_fit = fit
            lane.allx = fitx
            lane.bestx = np.mean(fitx)
            lane.radius_of_curvature = curved
            lane.current_fit = fit
        # If sanity check fails use the previous values
        else:
            lane.detected = False
            fitx = lane.allx
    else:
        # If lane was not detected and no curvature is defined
        if lane.radius_of_curvature:
            if abs(curved / lane.radius_of_curvature - 1) < 1:
                lane.detected = True
                lane.current_fit = fit
                lane.allx = fitx
                lane.bestx = np.mean(fitx)
                lane.radius_of_curvature = curved
                lane.current_fit = fit
            else:
                lane.detected = False
                fitx = lane.allx
                # If curvature was defined
        else:
            lane.detected = True
            lane.current_fit = fit
            lane.allx = fitx
            lane.bestx = np.mean(fitx)
            lane.radius_of_curvature = curved
    return fitx


# Sanity check for the direction
def sanity_check_direction(right, right_pre, right_pre2):
    # If the direction is ok then pass
    if abs((right - right_pre) / (right_pre - right_pre2) - 1) < .2:
        return right
    # If not then compute the value from the previous values
    else:
        return right_pre + (right_pre - right_pre2)


# find_lanes function will detect left and right lanes from the warped image.
# 'n' windows will be used to identify peaks of histograms
def find_lanes(n, image, x_window, lanes, left_lane_x, left_lane_y, right_lane_x, right_lane_y, window_ind):
    # 'n' windows will be used to identify peaks of histograms
    # Set index1. This is used for placeholder.
    index1 = np.zeros((n + 1, 2))
    index1[0] = [300, 1100]
    index1[1] = [300, 1100]
    # Set the center
    center = 700
    # Set the direction
    direction = 0
    for i in range(n - 1):
        # set the window range.
        y_window_top = 720 - 720 / n * (i + 1)
        y_window_bottom = 720 - 720 / n * i
        # If left and right lanes are detected from the previous image
        if (left_lane.detected == False) and (right_lane.detected == False):
            # Find the histogram from the image inside the window
            left = find_peaks(image, y_window_top, y_window_bottom, index1[i + 1, 0] - 200, index1[i + 1, 0] + 200)
            right = find_peaks(image, y_window_top, y_window_bottom, index1[i + 1, 1] - 200, index1[i + 1, 1] + 200)
            # Set the direction
            left = sanity_check_direction(left, index1[i + 1, 0], index1[i, 0])
            right = sanity_check_direction(right, index1[i + 1, 1], index1[i, 1])
            # Set the center
            center_pre = center
            center = (left + right) / 2
            direction = center - center_pre
        # If both lanes were detected in the previous image
        # Set them equal to the previous one
        else:
            left = left_lane.windows[window_ind, i]
            right = right_lane.windows[window_ind, i]
        # Make sure the distance between left and right laens are wide enough
        if abs(left - right) > 600:
            # Append coordinates to the left lane arrays
            left_lane_array = lanes[(lanes[:, 1] >= left - x_window) & (lanes[:, 1] < left + x_window) &
                                    (lanes[:, 0] <= y_window_bottom) & (lanes[:, 0] >= y_window_top)]
            left_lane_x += left_lane_array[:, 1].flatten().tolist()
            left_lane_y += left_lane_array[:, 0].flatten().tolist()
            if not math.isnan(np.mean(left_lane_array[:, 1])):
                left_lane.windows[window_ind, i] = np.mean(left_lane_array[:, 1])
                index1[i + 2, 0] = np.mean(left_lane_array[:, 1])
            else:
                index1[i + 2, 0] = index1[i + 1, 0] + direction
                left_lane.windows[window_ind, i] = index1[i + 2, 0]
            # Append coordinates to the right lane arrays
            right_lane_array = lanes[(lanes[:, 1] >= right - x_window) & (lanes[:, 1] < right + x_window) &
                                     (lanes[:, 0] < y_window_bottom) & (lanes[:, 0] >= y_window_top)]
            right_lane_x += right_lane_array[:, 1].flatten().tolist()
            right_lane_y += right_lane_array[:, 0].flatten().tolist()
            if not math.isnan(np.mean(right_lane_array[:, 1])):
                right_lane.windows[window_ind, i] = np.mean(right_lane_array[:, 1])
                index1[i + 2, 1] = np.mean(right_lane_array[:, 1])
            else:
                index1[i + 2, 1] = index1[i + 1, 1] + direction
                right_lane.windows[window_ind, i] = index1[i + 2, 1]
    return left_lane_x, left_lane_y, right_lane_x, right_lane_y


# Function to find the fitting lines from the warped image
def fit_lanes(image):
    # define y coordinate values for plotting
    yvals = np.linspace(0, 100, num=101) * 7.2  # to cover same y-range as image
    # find the coordinates from the image
    lanes = np.argwhere(image)
    # Coordinates for left lane
    left_lane_x = []
    left_lane_y = []
    # Coordinates for right lane
    right_lane_x = []
    right_lane_y = []

    # Find lanes from three repeated procedures with different window values
    left_lane_x, left_lane_y, right_lane_x, right_lane_y \
        = find_lanes(4, image, 25, lanes,
                     left_lane_x, left_lane_y, right_lane_x, right_lane_y, 0)
    left_lane_x, left_lane_y, right_lane_x, right_lane_y \
        = find_lanes(6, image, 50, lanes,
                     left_lane_x, left_lane_y, right_lane_x, right_lane_y, 1)
    left_lane_x, left_lane_y, right_lane_x, right_lane_y \
        = find_lanes(8, image, 75, lanes,
                     left_lane_x, left_lane_y, right_lane_x, right_lane_y, 2)
    # Find the coefficients of polynomials
    left_fit = np.polyfit(left_lane_y, left_lane_x, 2)
    left_fitx = left_fit[0] * yvals ** 2 + left_fit[1] * yvals + left_fit[2]
    right_fit = np.polyfit(right_lane_y, right_lane_x, 2)
    right_fitx = right_fit[0] * yvals ** 2 + right_fit[1] * yvals + right_fit[2]
    # Find curvatures
    left_curverad = find_curvature(yvals, left_fitx)
    right_curverad = find_curvature(yvals, right_fitx)
    curvatures = (left_curverad + right_curverad) / 2
    # Sanity check for the lanes
    left_fitx = sanity_check(left_lane, left_curverad, left_fitx, left_fit)
    right_fitx = sanity_check(right_lane, right_curverad, right_fitx, right_fit)

    return yvals, left_fitx, right_fitx, left_lane_x, left_lane_y, right_lane_x, right_lane_y, curvatures


def draw_poly(image, warped, yvals, left_fitx, right_fitx, Minv, curvature):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, new_warp, 0.3, 0)
    # Put text on an image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Radius of Curvature: {} m".format(int(curvature))
    cv2.putText(result, text, (400, 100), font, 1, (255, 255, 255), 2)
    # Find the position of the car
    pts = np.argwhere(new_warp[:, :, 1])
    position = find_position(pts, image.shape)
    if position < 0:
        text = "Vehicle is {:.2f} m left of center".format(-position)
    else:
        text = "Vehicle is {:.2f} m right of center".format(position)
    cv2.putText(result, text, (400, 150), font, 1, (255, 255, 255), 2)
    return result

ret, mtx, dist, rvecs, tvecs = calibrate(camera_cal_dir)


def draw_lane_lines(image):
    # Apply pipeline to the image to create black and white image
    img = pipeline(image)
    # Warp the image to make lanes parallel to each other
    top_down, perspective_M, perspective_Minv = corners_unwarp(img, mtx, dist)
    # Find the lines fitting to left and right lanes
    a, b, c, lx, ly, rx, ry, curvature = fit_lanes(top_down)
    # Return the original image with colored region
    return draw_poly(image, top_down, a, b, c, perspective_Minv, curvature)


if __name__ == '__main__':
    # testing image processor
    image = cv2.imread('test.jpg')
    result = draw_lane_lines(image)
    cv2.imwrite('result.png', result)
