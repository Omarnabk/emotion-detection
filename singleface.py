import argparse

import cv2
import keyboard
import numpy as np

from em_model import EMR

EMOTIONS = ['Enfado', 'Asco', 'Temor', 'Felicidad', 'Tristeza', 'Sorpresa',
            'Neutral']  # ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

# initialize the cascade
cascade_classifier = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')


def format_image(image):
    """
    Function to format frame
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
        # determine whether the image is color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Image read from buffer
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)

    if not len(faces) > 0:
        return None

    # initialize the first face as having maximum area, then find the one with max_area
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face

    # extract ROI of face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    try:
        # resize the image so that it can be passed to the neural network
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("----->Problem during resize")
        return None

    return image


def main(cam_idx):
    full_screen = True

    # Initialize object of EMR class
    network = EMR()
    network.build_network()

    cap = cv2.VideoCapture(cam_idx)
    font = cv2.FONT_HERSHEY_SIMPLEX
    feelings_faces = []

    # append the list with the emoji images
    for index, emotion in enumerate(EMOTIONS):
        feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))
    facecasc = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
    while True:
        # Again find haar cascade to draw bounding box around face

        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, 1.3, 5)

        # compute softmax probabilities
        result = network.predict(format_image(frame))
        if result is not None:
            # write the different emotions and have a bar to indicate probabilities for each class
            # for index, emotion in enumerate(EMOTIONS):
            #     cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1);
            #     cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
            #                   (255, 0, 0), -1)

            # find the emotion with maximum probability and display it
            maxindex = np.argmax(result[0])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, EMOTIONS[maxindex], (10, 360), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
            face_image = feelings_faces[maxindex]
            # print(face_image[:, :, 3])

            for c in range(0, 3):
                # the shape of face_image is (x,y,4)
                # the fourth channel is 0 or 1
                # in most cases it is 0, so, we assign the roi to the emoji
                # you could also do:
                # frame[200:320,10:130,c] = frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)
                frame[200:320, 10:130, c] = face_image[:, :, c] * (face_image[:, :, 3] / 255.0) + frame[200:320, 10:130,
                                                                                                  c] * (
                                                    1.0 - face_image[:, :, 3] / 255.0)

        if not len(faces) > 0:
            # do nothing if no face is detected
            a = 1
        else:
            # draw box around face with maximum area
            max_area_face = faces[0]
            for face in faces:
                if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
                    max_area_face = face
            face = max_area_face
            (x, y, w, h) = max_area_face
            frame = cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)

        if cv2.waitKey(1) & keyboard.is_pressed('f'):
            full_screen = True

        if full_screen:
            cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow("window", cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--cam", help="Camera Index", default=0)
    args = argParser.parse_args()
    cam_idx = args.cam
    main(cam_idx=int(cam_idx))
