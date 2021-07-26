import face_recognition
from face_recognition.api import face_encodings


# Training

train_encodings=[]
train_names=[]

train_image = face_recognition.load_image_file("images/train.jpg")
train_face_patch = face_recognition.face_locations(train_image, model="hog")

train_face_encodings=face_recognition.face_encodings(train_image, train_face_patch)

train_encodings.append(train_face_encodings[0])
train_names.append("bilal")



# Testing
test_image = face_recognition.load_image_file("images/train.jpg")
test_face_patches = face_recognition.face_locations(test_image, model="hog")
test_face_encodings=face_recognition.face_encodings(test_image, test_face_patches)

for encoding in test_face_encodings:
    results = face_recognition.compare_faces(train_encodings, encoding)
    if True:
        index=results.index(True)
        print(train_names[index], "exists")

