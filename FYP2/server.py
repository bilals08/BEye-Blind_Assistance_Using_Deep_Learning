
from flask import json, request, send_file, render_template
from flask.wrappers import Response
from model.fchardnet import hardnet
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import base64
from flask_cors import CORS
from datetime import datetime
import time
import pyttsx3
import threading
from io import BytesIO

from flask import Flask,jsonify


#import feedback
#============================ INITILIAZING GLOBAL MEMBERS ================================
app = Flask(__name__, template_folder='template')
cors = CORS(app)

camera = cv2.VideoCapture("video.mp4") #Loading demo video
camera_depth = cv2.VideoCapture("depth.avi") #Loading demo video
time.sleep(3)
global index_add_counter
index_add_counter=0



train_id_to_color=np.array([(225 ,229 ,204),
 ( 70 , 70 , 70),
 (152, 152 , 79),
 ( 70 ,130, 180),
 (244 , 35 ,232),
 (152 ,251 ,152),
 (107 ,142 , 35),
 (220 , 20 , 60),
 (139 ,218 , 51),
 ( 41 , 34 ,177),
 (111 , 34 ,177),
 (211 ,205 , 33),
 (147 ,147 ,136),
 (241 ,170,  17),
 ( 29 ,231 ,229),
 ( 35 , 18 , 16),
 ( 51 ,  0 ,  0),
 (  0 ,  0 ,  0)])


allLabels= {
    0 : 'background',
        1 : 'wall',
        2 : 'building',
        3 : 'sky',
        4 : 'sidewalk',
        5: 'field',
        6 : 'vegitation',
        7 : 'person',
        8 : 'mountain',
        9 : 'car',
        10: 'bike',
        11: 'animal',
        12: 'ground',
        13: 'fence',
        14: 'water',
        15: 'road',
        16: 'ceiling',
    
    }
#=========================================================================================


def howNear(value):
    ans=""
    if value>200:
        ans="Too Near"
    elif value>150:
        ans="close"
    else:
        ans="far"
    return ans


def threadingWindow(say_text):
    #print("djasjd")
    engine = pyttsx3.init()
    engine.setProperty('rate', 400)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say(say_text)
    #engine.startLoop(False)
    engine.runAndWait()
    engine.endLoop()
    

def GetFeedback(mask1D, maskH):
    
    pavementLabels=[4,5,12,15]
    obstacleLabels=[1,2,7,9,10,13]
    otherLabels=[0,3,14,16,6,8]
    
    targetArea= mask1D[200:,200:500]
    targetmaskH = maskH[200:,200:500]
    

    uniqueValues, counts=np.unique(targetArea,return_counts=True) ## returning Unique Values in the region and which targets have max pixels
    uv=[]
    co=[]
    for index, (value, count) in enumerate(zip(uniqueValues, counts)): ## Removing targets with less than 50 pixels
        
        if count < 500:
            continue
        
        uv.append(value)
        co.append(count)
        
    uniqueValues=np.array(uv)
    counts=np.array(co)
    
    # print(uniqueValues)
    # print(counts)       
            

    sortedIndices=np.argsort(-counts) ## descending order sorting

    # print(uniqueValues[sortedIndices[0]]) ## Fetching Max Value
    closestObstacle=""

    maxDepth=np.max(targetmaskH)
    closeObstaclesDepth=np.argwhere(targetmaskH==maxDepth)
    res=list(map(tuple, closeObstaclesDepth))
    closeObstacle=np.array([targetArea[i] for i in res])
    obsU, obsC = np.unique(closeObstacle, return_counts=True)
    
    obsSorted=np.argsort(-obsC)
    closestObstacle=""
    for i in obsSorted:
        if obsU[i] in obstacleLabels:
            closestObstacle=obsU[i]
            break
             

    pavement=False
    obstacleCount=0
    pavementFeedback=[]
    obstacleFeedback=[]
    feedbackLabels=[]
    if closestObstacle!="":
        obstacleFeedback.append(allLabels[closestObstacle])
        feedbackLabels.append(allLabels[closestObstacle] + " " +howNear(maxDepth))
    for i in range(sortedIndices.shape[0]):
        
        count= sortedIndices[i]
        label = uniqueValues[i]
        
        if label in pavementLabels and not pavement: ##Only one pavement label at one time
            pavement=True
            feedbackLabels.append(allLabels[label])
            pavementFeedback.append(allLabels[label])
            
        
        if label in obstacleLabels and label!=closestObstacle:
            if(obstacleCount<2):
                feedbackLabels.append(allLabels[label])
                obstacleFeedback.append(allLabels[label])
                obstacleCount+=1
   
    say_text=""
    for label in pavementFeedback:
        # engine.setProperty('rate', 200)
        say_text+="On "+label
        #engine.say("walking on "+ label )
        #engine.runAndWait()
    
    for index, label in enumerate(obstacleFeedback):
        if index==0 and closestObstacle!="":
            say_text+=" "+label +" "+ howNear(maxDepth)
            #engine.say(label + howNear(maxDepth))
            #engine.runAndWait()
        else:
            say_text+=" "+label+" ahead"
            #engine.say(label + "ahead")
            #engine.runAndWait()

    # engine.iterate() must be called inside Server_Up.start()
    # Server_Up = threading.Thread(target = )

    Server_Up= threading.Thread(target=threadingWindow, args=(say_text,))
    Server_Up.start()

    # engine.endLoop()
    print(feedbackLabels)

    return feedbackLabels


def transform_image(image):
    my_transforms = transforms.Compose([
        transforms.Resize((480,640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    image=Image.fromarray(image)
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor)[0]
    output_predictions = outputs.argmax(0)
    return output_predictions

def decode_target(target):
    return train_id_to_color[target]

def passImage(image,depth):
    output_prediction = get_prediction(image)
    img=decode_target(np.array(output_prediction.cpu())).astype(np.uint8)
    img=img[:,:,::-1]
    val=GetFeedback(np.array(output_prediction),depth)
    #print(val)
    return img,val

def to_image(numpy_img):
    img = Image.fromarray(numpy_img, 'RGB')
    return img

def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "PNG") 
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8')

def gen_frames1():
    success, frame = camera.read() 
    success, depth = camera_depth.read()
    # print("Depth",depth.shape)
    gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    # print("Gray", np.amax(gray))
    # print("Frame",frame.shape)
    # index_add_counter+=1
    # print(index_add_counter)
    if not success or not success:
        return {}
    fr,labels=passImage(frame,gray.astype(np.uint8))
    frame1 = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    pil_image1 = to_image(frame1)
    image_uri1 = to_data_uri(pil_image1)

    frame2 = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
    pil_image2 = to_image(frame2)
    image_uri2 = to_data_uri(pil_image2)

    data = json.dumps({'frame1':image_uri1,'frame2':image_uri2, 'labels':labels})
    return data
  

@app.route('/video_feed', methods = ['GET'])    
def video_feed():
    return gen_frames1()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live_feed')
def render_feed():
    return render_template('video_feed.html')

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
model = hardnet(17)
ckpt="model/last_model.pth"
checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

#============================================================================================

if __name__ == "__main__":
    app.run(debug=True)