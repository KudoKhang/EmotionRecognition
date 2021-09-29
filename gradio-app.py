import cv2
import gradio as gr
import numpy as np 
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")


face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('files/model_10epoch_val_acc_0.638.h5', compile=False)
EMOTIONS = ["Tuc gian" ,"Kinh tom","So hai", "Hanh phuc", "Buon ba", "Bat ngo", "Binh thuong"]

def classify(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30,30))
    
    # Chỉ thực hiện nhận biết cảm xúc khi phát hiện được có khuôn mặt trong hình
    if len(faces) > 0:
        # Chỉ thực hiện với khuôn mặt chính trong hình (khuôn mặt có diện tích lớn nhất)
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        # Tách phần khuôn mặt vừa tìm được và resize về kích thước 48x48, vì mạng mình train có đầu vào là 48x48
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]
        return label
        # print(label)

# img = cv2.imread('./files/1.jpg')
# classify(img)

iface = gr.Interface(
    classify, 
    gr.inputs.Image(shape=(224, 224)),
    # gr.inputs.Image(source="webcam", shape=(224,224), tool=None), 
    gr.outputs.Label(),
    capture_session=True,
    examples=[
        ["./files/1.jpeg"],
        ["./files/2.png"],
        ["./files/3.jpeg"],
        ["./files/4.jpg"],
        ["./files/5.jpeg"],
        ["./files/6.jpg"],
        ["./files/7.jpg"],
        ["./files/8.jpg"],
    ])

if __name__ == "__main__":
    iface.launch(share=True)