import os
from glob import glob
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import FaceAttributeType
from msrest.authentication import CognitiveServicesCredentials

face_attributes = []
for name, member in FaceAttributeType.__members__.items():
    attr = member.value
    face_attributes.append(attr)
print(face_attributes)

KEY = '4a26c64150d244a5ab05fa947f78ca2f'
ENDPOINT = 'https://jengzhu8.cognitiveservices.azure.com/'
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
images = sorted(glob('/home/zhu/idinvert/examples/*.png'))

for image in images:
    img = open(image, 'rb')
    detected_faces = face_client.face.detect_with_stream(image=img, return_face_attributes=face_attributes)
    if not detected_faces:
        raise Exception(
            'No face detected from image {}'.format(os.path.basename(image)))

    # Face IDs are used for comparison to faces (their IDs) detected in other images.
    for face in detected_faces:
        print()
        print('Detected face ID from', os.path.basename(image), ':')
        print(face.face_id)
        print()
        print('Facial attributes detected:')
        print('Age:', face.face_attributes.age)
        print('Gender:', face.face_attributes.gender)
        print('Head pose:', face.face_attributes.head_pose)
        print('Smile:', face.face_attributes.smile)
        print('Facial hair:', face.face_attributes.facial_hair)
        print('Glasses:', face.face_attributes.glasses)
        print('Emotion:', face.face_attributes.emotion)
        print('hair:', face.face_attributes.hair)
        print('makeup:', face.face_attributes.makeup)
        print('occlusion:', face.face_attributes.occlusion)
        print('accessories:', face.face_attributes.accessories)
        print('blur:', face.face_attributes.blur)
        print('exposure:', face.face_attributes.exposure)
        print('noise:', face.face_attributes.noise)
        print()