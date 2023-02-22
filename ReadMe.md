
<img src="ReadMe%20Github%2037de4cc43ebc44a7bf7e1f356c3eeae3/Untitled.png" alt="drawing"/>

# Introduction

Our project aims to assist the blind community in navigation for outdoors through the use of state of the art Deep Learning and Computer Vision techniques. It will help that person by providing terrain awareness (e.g road, sidewalk, ground etc) and helping detect obstacles in the way (e.g person, car, bike, etc). We shall combine the output of our model with our RGB-D sensor where the semantic map from the model will be combined with depth map from the sensor to give out real-time audio alerts to the user for further awareness.

# Approach & Models

We have trained 3 models to opt for the best one.

1. DeelpabV3+ with MobileNet backbone
2. U-FCHarDnet-70
3. Unet with Efficient Net [ b0 ]

FCHarDNet outperforms DeeplabV3+ and Unet for mIOU and Forward Pass Time based on our research.

![ReadMe%20Github%2037de4cc43ebc44a7bf7e1f356c3eeae3/Capture6.png](ReadMe%20Github%2037de4cc43ebc44a7bf7e1f356c3eeae3/Capture6.png)

<img src="ReadMe%20Github%2037de4cc43ebc44a7bf7e1f356c3eeae3/Capture7.png" alt="drawing" width="50%" height="40%"/>

All the models were trained and finetuned on ADE-20K Screen Parsing, COCO stuff, and Custom Sidewalk datasets for 17 classes including terrain awareness (Sidewalk & Floor, Road, Ground), obstacle detection, sky and building detection. The highlighted ones from the above table show the best performing ones.

## Stages Done :

Below are the stages done illustrated by 'Done' and rest are our future work.

1. **Stage #1:** Extracting the RGB-D Image from Sensor
2. **Stage #2:** Model Setup and Predictions 
3. **Stage #3:** Combining RGB Image with Depth Map & Giving Real-time Alerts (Obstacle Avoidance, Person Identification)
4. **Stage #4:**  Final Refined Working Product

# Guidelines

## Requirements :

numpy==1.18.1

matplotlib==3.1.3 

torch==1.6.0

pandas==1.0.1

pip==20.2.4

tensorboard==2.3.0

tqdm==4.42.1

## Usage :

To use a model for training or prediction, place the utils files in the model's notebook folder. 

## Training :

For training, call Train Model function in the notebook after running Utility, Model , DataLoader and Training Setup in the notebook.

```markdown
TrainModel(imageFolders,targetFolders,learningRate=0.00001,weightDecay=0.0001,learningRatePolicy='poly', noOfEpochs=27,stepSize=10000, savedPath="/",pretrained=False, batchSize=16,valBatchSize=16, lossFunction="focal", useCuda=True):
```

To train a function, you must give the path of the folders which contains both training and validation folder in it.

- imageFolders :  list of paths of Images
- targetFolders : list of paths of Groundtruth
- noOfEpochs:  epochs
- if pretrained=True then you must give savedPath=path for a pretrained weights
- if GPU is available,use useCuda=True
- Use batchSize for training batch size
- Use valBatchSize for validation batch size
- loss function could be 'focal' or 'entropy'
- 255 is always the ignore index, which means it will not calculate the loss of 255 label in ground truth
- Your ground truth must have labels 0 to 16, and it can have 255 to ignore it

## Prediction On Image :

For prediction, call PredictImage function in the notebook after running Utility, Model and Prediction Setup in the notebook.

```markdown
PredictImage(input_image, useCuda=True,num_classes=17,pretrained=False, saved_path="/" )
```

- input_image: path of the image to predict
- pretrained: True if model has pretrained weights
- saved_path : path of pretrained weigths
- useCuda: True if GPU is available
- num_classes: no of classes your model is trained upon, default is 17.

## Prediction On Video :

For prediction on Video, call PredictVideo function in the notebook after running Utility, Model and Prediction Video Setup in the notebook.

```markdown
PredictVideo(video_path,useCuda=True,num_classes=17,pretrained=False, num_frames = 20000 ,saved_path="result",output_path="results"):
```

- video_path: path of the video to predict
- pretrained: True if model has pretrained weights
- saved_path : path of pretrained weigths
- useCuda: True if GPU is available
- num_classes: no of classes your model has
- num_frames: frames process from the entire video
- output path: path of resultant output (.mp4)

## Prediction Results :

The results of our models after training and finetuning have been illustrated in this section. The predictions show us where the noise is present and which model predicts boundaries well. Below are the visualizations:

### a. U-FCHarDNet-70 :

![ReadMe%20Github%2037de4cc43ebc44a7bf7e1f356c3eeae3/Capture.png](ReadMe%20Github%2037de4cc43ebc44a7bf7e1f356c3eeae3/Capture.png)

***Prediction:***

![ReadMe%20Github%2037de4cc43ebc44a7bf7e1f356c3eeae3/Capture2.png](ReadMe%20Github%2037de4cc43ebc44a7bf7e1f356c3eeae3/Capture2.png)

### **b. DeepLab V3+ :**

![ReadMe%20Github%2037de4cc43ebc44a7bf7e1f356c3eeae3/Capture3.png](ReadMe%20Github%2037de4cc43ebc44a7bf7e1f356c3eeae3/Capture3.png)

***Prediction:***

![ReadMe%20Github%2037de4cc43ebc44a7bf7e1f356c3eeae3/Capture4.png](ReadMe%20Github%2037de4cc43ebc44a7bf7e1f356c3eeae3/Capture4.png)

### **c. UNet with EfficientNet [ b0 ]:**

![ReadMe%20Github%2037de4cc43ebc44a7bf7e1f356c3eeae3/Capture5.png](ReadMe%20Github%2037de4cc43ebc44a7bf7e1f356c3eeae3/Capture5.png)

***Prediction:***

The size of the UNET with Efficient Net-b0 became too large and it took around 200 ms in forward pass of a single image as well as having mean IoU of 12.5% (on 110 epochs) meaning not suitable for the application hence results were excluded.

### Authors

Feel free to contact us for further details.

**Muhammad Bilal Shabbir**
 *bilal.shabbir@nu.edu.pk*
 *FAST NUCES, Islamabad*

**Sohaib Akhtar**
*i170330@nu.edu.pk*
*FAST NUCES, Islamabad*

**Ali Salman**
*i170350@nu.edu.pk*
*FAST NUCES, Islamabad*

