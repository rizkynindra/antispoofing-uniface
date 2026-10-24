import cv2
from uniface.detection import RetinaFace
from uniface.spoofing import MiniFASNet

detector = RetinaFace()
spoofer = MiniFASNet()

image = cv2.imread("assets/test_images/382A5649.jpg")
faces = detector.detect(image)

for face in faces:
    result = spoofer.predict(image, face.bbox)

    label = "Real" if result.is_real else "Fake"
    print(f"{label}: {result.confidence:.1%}")