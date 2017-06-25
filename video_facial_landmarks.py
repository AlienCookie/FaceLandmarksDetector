# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
import math
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import datetime
import argparse
import imutils
import time
import dlib
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

def rotate_image(image, angle):

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def demo():

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        leftP = None
        rightP = None

        rows, cols = frame.shape[:2]

        rectangle = {'left' : (0,0), 'right' : (0,0), 'bottom' : (0,0), 'top' : (0,0)}

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks
            # caculate bounding rectangle and draw them on the image
            i = 0
            for (x, y) in shape:
                if(i == 36):
                    leftP = (x, y)
                    cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
                elif (i == 45):
                    rightP = (x, y)
                    cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
                else:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                if i == 0:
                    rectangle['left'] = (x, y)
                elif i == 8:
                    rectangle['bottom'] = (x, y)
                elif i == 16:
                    rectangle['right'] = (x, y)
                elif i == 24:# or i == 19:
                  #  if rectangle['top'] == None or rectangle < y:
                    rectangle['top'] = (x, y)
                i = i + 1
            break

        # calculate head angle
        if(leftP == None or rightP == None):
            angle = 0
        else:
            P12 = math.fabs(leftP[0] - rightP[0])
            P13 = math.sqrt((leftP[0] - rightP[0])**2 + (leftP[1] - rightP[1])**2)
            P23 = math.fabs(leftP[1] - rightP[1])
            if(P12 == 0 or P13 == 0 or P23 == 0):
                angle = 0
            else:
                angle = math.acos((P12**2 + P13**2 - P23**2) / (2 * P12 * P13)) * 180.0 / math.pi * (rightP[1] - leftP[1]) / P23

        centralizedLeft = (rectangle['left'][0] - cols / 2, rectangle['left'][1] - rows / 2)
        centralizedRight = (rectangle['right'][0] - cols / 2, rectangle['right'][1] - rows / 2)
        centralizedBottom = (rectangle['bottom'][0] - cols / 2, rectangle['bottom'][1] - rows / 2)
        centralizedTop = (rectangle['top'][0] - cols / 2, rectangle['top'][1] - rows / 2)

        # rotate face rectangle
        recLeft   = int(centralizedLeft[0] * math.cos(angle) - centralizedLeft[1] * math.sin(angle) + cols/2)
        recRight  = int(centralizedRight[0] * math.cos(angle) - centralizedRight[1] * math.sin(angle) + cols/2)
        recBottom = int(centralizedBottom[0] * math.sin(angle) + centralizedBottom[1] * math.cos(angle) + rows/2)
        recTop    = int(centralizedTop[0] * math.sin(angle) + centralizedTop[1] * math.cos(angle) + rows/2)

        # print points and angle
        print(leftP, rightP, angle)

        M = cv2.getRotationMatrix2D((int((recRight + recLeft)/2), int((recBottom + recTop)/2)), angle, 1)
        image_rotated = cv2.warpAffine(frame, M, (cols, rows))

        cv2.circle(image_rotated, (int((recRight + recLeft)/2), int((recBottom + recTop)/2)), 6, (0, 255, 255), -1)
        cv2.rectangle(image_rotated, (recLeft, recTop), (recRight, recBottom), (255, 0, 255), 4, 8, 0)
        '''cv2.circle(image_rotated, (recLeft, int(rows/2)),     6, (0, 255, 255), -1)
        cv2.circle(image_rotated, (recRight,int(rows/2)),     6, (0, 255, 255), -1)
        cv2.circle(image_rotated, (int(cols/2), recBottom),   6, (0, 255, 255), -1)
        cv2.circle(image_rotated, (int(cols/2), recTop),      6, (0, 255, 255), -1)'''


        '''gray_rotated  = cv2.warpAffine(gray, M, (cols, rows))

        rects = detector(gray_rotated, 0)

        lowP = None
        leftTopP = None
        rightTopP = None

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray_rotated, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            i = 0
            for (x, y) in shape:
                if (i == 8):
                    lowP = (x, y)
                elif (i == 0):
                    leftTopP = (x, y)
                elif (i == 16):
                    rightTopP = (x, y)
                i = i + 1

            break

        image_rotated = rotate_image(frame, angle)
        image_rotated_cropped = crop_around_center(
            image_rotated,
            *largest_rotated_rect(
                cols,
                rows,
                math.radians(angle)
            )
        )

        if( lowP != None and leftTopP != None  and rightTopP != None):
            image_rotated_cropped = image_rotated[leftTopP[0] : lowP[1],
                                    rightTopP[0] - leftTopP[0] : rightTopP[1] - lowP[1]]
        else:
            image_rotated_cropped = image_rotated'''



        # show the frame
        cv2.imshow("Frame", image_rotated)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    demo()
