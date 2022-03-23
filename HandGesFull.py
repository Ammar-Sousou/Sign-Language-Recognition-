import cv2
import numpy as np
from gtts import gTTS 
from gingerit.gingerit import GingerIt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras as ks

RES = 224
SIZE = (RES, RES)
UPPER_LEFT = (376, 82)
BOTTOM_RIGHT = (600, 306)
text_list = []
MODEL_PATH = r"G:\Users\new LAPTOP\Desktop\Desktop\eng\5th\GP\models\ASL_Model.h5"
MODEL = ks.models.load_model(MODEL_PATH, compile=False)
LABELS = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
    26: "delete",
    27: "nothing",
    28: " "
}


def normalize_input(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, SIZE)
    image_array = image.reshape(1, RES, RES, 3) / 255.0
    return image_array


def get_prediction(image_array: np.ndarray) -> str:
    # Use the model to predict the sign
    prediction = MODEL.predict(image_array)

    # Get the accuracy of the detected sign
    accuracy = prediction[0][np.argmax(prediction[0])]

    # Only print if the accuracy is larger than 60%
    if float(accuracy) > 0.6:
        return LABELS.get(np.argmax(prediction[0]))


def quit_condition(*args: int) -> bool:
    return True if cv2.waitKey(1) & 0xFF in args else False


def text_maker(*args: int) -> bool:
    if cv2.waitKey(1) & 0xFF in args:
        
        # Delete the last character from the text
        if get_prediction(image_array) == 'delete':
            del text_list[-1]
            print("The text has been formed so far:", ''.join(j for j in text_list))
        
        # Capitalize the character if it is at the beginning of the text
        else:
            if len(text_list) == 0:
                text_list.append(get_prediction(image_array))
                print("The text has been formed so far:", ''.join(j for j in text_list))
            else:
                text_list.append(get_prediction(image_array).lower())
                print("The text has been formed so far:", ''.join(j for j in text_list))
                
            

def speaker_fun(txt):
    # Linguistic correction and Pronunciation of the formed text
    myobj = gTTS(text = txt, lang = 'en', slow = True) 
    myobj.save("text.mp3") 
    os.system("start text.mp3")


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        # Read and flip the video frame
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Draw the ROI
        cv2.rectangle(frame, UPPER_LEFT, BOTTOM_RIGHT,
                        color=(100, 50, 200), thickness=5)

        # Show full image
        cv2.imshow("Full Image", frame)

        # Slice and show sliced image
        sliced_image = frame[82:306, 376:600]
        cv2.imshow("Sliced Image", sliced_image)

        # Flip and show the flipped sliced image
        sliced_image = cv2.flip(sliced_image, 1)
        cv2.imshow("Flipped Sliced Image", sliced_image)

        # Normalize the flipped and sliced image
        image_array = normalize_input(sliced_image)

        # Predict the sign form the image and print the result
        prediction = get_prediction(image_array)
        if prediction is not None:
            print(prediction)

        # Make a text by collecting the letters 
        text_maker(13, 32)
        
        # Quit and speak if a quit condition is met
        if quit_condition(ord('q'), 27):
            after_join = ''.join(j for j in text_list)
            print('--------------')
            if not after_join:
                print("You didn't enter text!")
            else:
                print("Text that was formed:",GingerIt().parse(after_join)['result'])
                speaker_fun(GingerIt().parse(after_join)['result'])
            break


cap.release()
cv2.destroyAllWindows()