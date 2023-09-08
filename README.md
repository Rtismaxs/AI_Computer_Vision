# AI_Computer_Vision
This repository is dedicated to AI implementation for Computer Vision using GPUs for training  

############################## README FILE #############################
TRAINED MODELS ARE STORED IN THE KAGGLE DATASET “Models Trained S383430 MATHIS CADIER”, BECAUSE THEY ARE TOO HEAVY TO BE STORED IN THE ZIP FILE

MOREOVER ALL DATA USED THROUGH NOTEBOOKS (ZIP AND CSV FILES) ARE ALSO STORED IN THE KAGGLE DATASET “NOTEBOOK DATA S383430 MATHIS CADIER”

First of all, because there is 200 GB of data (either for training and test), you should start a Kaggle Notebook directly from the Competition window in order to have those data already included in your Notebook.

1) Visualisation and Pre-Processing

So, the first Notebook that you should run on Kaggle is the 'Vinbigdata_Visualisation_Preprocessing.ipynb'. Normally, you should have access to the 'VinBigData Chest X-Ray Abnormalities Detection' dataset containing two folders 'train' and 'test' filled with DICOM images and two csv files 'sample_submission.csv' and 'train.csv'. 

1.1) Visualisation

Now that everything has been setup, you can run all the cells from the Notebook. Indeed, first we import all libraries needed in this Notebook. Then, we read both csv files using 'Pandas'. We looked into more details (some statistics) for the dataframe coming from the 'train.csv', because it will be the one, we will the most work on. After that, we plot first visualisations using 'Seaborn' library:
- A catplot to count in total the number of each abnormality
- A catplot to count in total the number of diagnoses realised for each radiologist
- A catplot to count in total the number of each abnormality diagnosed for each radiologist
  
Then, we split our main dataframe into 2 dataframes: one with images belonging to the 14 types of abnormalities and the other one with images belonging only to the "No Finding" class. In the dataframe with the images from the 14 classes, we plot 2 new figures similar to the previous ones:
- A catplot to count in total the number of each abnormality
- A catplot to count in total the number of each abnormality diagnosed for each radiologist

To finish the visualisation part, we display an X-Ray chest image using 'Pydicom' that converts DICOM image into an array and using 'Matplotlib' to show a figure.

1.2) Pre-Processing

Regarding the pre-processing, we create the function 'convert_dicom_to_jpg' that convert DICOM images to JPG images and stores the image height and width into a dictionary. We convert DICOM to JPG, because when we will train our model, we will need JPG images. Moreover, it is easier to manipulate JPG images instead of DICOM images. So, we convert DICOM images to JPG images that we save into folders which are then zipped (one for training, 'dataset-images-jpg.zip, and one for submission, 'dataset-submissionimages-jpg.zip', images). Like that we will be able to import our data (the size of the ZIP file for the training data is about 7GB) to work on Google Colab for instance. In addition, we save new version of our two csv files 'sample_submission.csv' and 'train.csv' adding the height and width of the image: we call them 'sample_submission_with_sizes.csv' and 'train_with_sizes.csv'. 

We just finished to go through the first Notebook 'Vinbigdata_Visualisation_Preprocessing.ipynb'! Now you have 2 choices:
> Open and run the Notebook 'Vinbigdata_More_Visualisation.ipynb' which contains more visualisations using the csv file 'train_with_sizes.csv' that we just created. These visualisations are more about the position of bounding boxes on an X-Ray chest image and diagnosed by which radiologists using again 'Seaborn' with its scatterplot and regression plot.
> Keep going into the pipeline of the solution and open the Notebook 
'VinBigData_ResNet18.ipynb' to see how we train the ResNet18 model.

In order to have a reasonable size of this file, we will keep going and open the Notebook 'VinBigData_ResNet18.ipynb', but feel free to open and run the 'Vinbigdata_More_Visualisation.ipynb' Notebook for more visualisations.

2) ResNet18
   
As before, the Notebook 'VinBigData_ResNet18.ipynb' starts with the import of libraries used within this Notebook. First, we unzip ZIP files previously created 'dataset-images-jpg.zip' and 'dataset-submission-images-jpg.zip' into specific folders 'images' and 'submissions'. Then, we import 2 dataframes: 'train.csv' and 'sample_submission_with_sizes.csv'. As we know, the ResNet18 will be used to classify images between 2 classes "Finding" and "No Finding". So, we need to pre-process our data.

2.1) ResNet18 Pre-Processing

We create a new dataframe, 'data_2_cls', which comes from the 'train.csv' where we classify the images between the 2 classes. Then, as we can see on the catplot, we have imbalance data. Then, we also pre-process the 'sample_submission_with_sizes.csv' in order to have the right format for the output of the ResNet18. Following this, we pre-process each image of our data set by changing their size to (256,256) and standardising them. We apply this pre-processing for the training and submission images.

We then divide our main dataframe into a training and validation dataframes using in addition the argument 'stratify' for imbalanced data. Moreover, 2 folders '/vinbigdata/images/train' and '/vinbigdata/images/val' are created to store the respective images. Using the 'training', 'validation' and 'submission' dataframes, we first create 3 PyTorch datasets. 

Each dataset contains images transformed to tensor and their classes. Then, using these 3 PyTorch datasets, we are able to create 3 PyTorch dataloaders with a batch size of 16. For the 'training' dataloader, we created a 'WeightedRandomSampler' because of the imbalance data. Finally, we display batch's images and their labels thanks to the 'imshow' function.

2.2) ResNet18 Training

For the training of the ResNet18, we use a GPU to accelerate the time of training. So, it implies that we need to on GPU memory the weights of the model and the inputs coming from dataloaders: to do so, we use the '.to(device)' syntax, where the device is the GPU we are using defined by 'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")'.

We start the training on a pre-trained version of the ResNet18. We train this model on 10 epochs, using the 'CrossEntropyLoss', the optimizer 'SGD' and the lr scheduler 'StepLR'. During each epoch, we give as input to our ResNet18 model data coming from the 'training' dataloader in the form of batch. For each batch, we calculate the accuracy and the loss of the 
model outputs and we update the weights of our model. After going through each batch, we validate our ResNet18 on the 'validation' dataloader where we calculate again the accuracy and the loss. At the end of an epoch, we display the average accuracy and loss of our ResNet18 model.We repeat the previous steps for 10 epochs and we save the model with the higher accuracy.

2.3) ResNet18 Evaluation

To see predictions made by our trained ResNet18, we can go through the 'validation' dataloader and display the images and their predictions and ground truths using the 'visualize_model' function. In addition, we display the confusion matrix and the ROC curve with its AUC, in order to evaluate the predictions of our ResNet18 model in general.

Finally, we process our ResNet18 model on the 'submission' dataloader in order to classify each image with a certain confident score. We save these predictions as a csv file 'submission_2_cls_resnet18.csv' and we also save our fine-tuned model as 'best_resnet18_pretrained_10.pt'. Now that we finished with the training of the ResNet18, we will open the Notebook 
'VinBigData_YOLOv5x.ipynb' in order to explain a little more bit more the training of our YOLOv5x model in details.

4) YOLOv5

Again, as before, we first import all the libraries and then unzip the ZIP files in order to have access to the JPG images. We also read both 'train_with_sizes.csv' and 'sample_submission_with_sizes.csv' files into dataframes.

3.1) YOLOv5 Pre-Processing

Like previously, we reshape and standardise 'training' and 'submission' images into a (640,640) dimension. Then, we perform the "WBF" method in order to delete redundant bounding boxes and we create a new dataframe with the new bounding box values (for 'x_min', 'y_min', 'x_max' and 'y_max'). From that dataframe, we calculate the middle of each box and we create an alternative dataframe which will be used for creating labels for each 'training' image. In this alternative dataframe, 'data_final', we have lists with all classes, all 'x_mid', all 'y_mid' and all 'width'/'height'. Using these lists, we are capable to create a txt file in which labels of a specific image will be written. 

After doing this, we split our main dataframe into a 'training' and 'validation' dataframes. This time, we use the "KFold" method that allocates a value between 0 to 4 for each image. Then, we select a value and all images with this value as fold will become our 'validation' data. As before, we create two separate folders for 'training' and 'validation' purpose. 

Finally, we generate a YAML file that will be used during the training of the YOLOv5x. This YAML needs to contain the names of each class, the total number of class and the path of two txt file: 'train.txt' containing training image paths and 'val.txt' containing validation image paths.

3.2) YOLOv5 Training

Regarding the training part, we first clone a GitHub repository in order to obtain the weights of our YOLOv5x and also to have access to needed files while training. We install extra packages and then, we train on GPU our YOLOv5x pre-trained version using the command line:
"!python train.py --img 640 --batch 32 --epochs 60 --data /content/vinbigdata.yaml --weights yolov5x.pt --cache"

It implies that we are using images with a size of (640,640), batch size of 32, 60 epochs, our YAML file as data, downloaded weights and cache memory. As for the ResNet18, while training, we are calculating different losses specific to image detection like 'box_loss', 'cls_loss' or 'obj_loss' and we are also calculating the precision, recall and mAP of our YOLOv5x for each epoch. Even on GPU, the training of the YOLOv5x is very long (3h30) and at the end, we obtain a summarization of the performance from our trained model. Then, for the testing of our YOLOv5x trained model, we also use a command line:
"!python detect.py --weights 'runs/train/exp/weights/best.pt' --img 640 --conf 0.15 --iou 0.45 --source /content/submission-640/ --name yolov5x_results --save-txt --save-conf --exist-ok"

We specify here that we take weights of our best version of the YOLOv5x previously trained and we will process images from the folder '/content/submission-640/' with images of dimension (640,640) with a confident threshold of 0.15 and a IoU threshold of 0.45. In addition, we save the outputs as txt files for each image thanks to the '--save-txt' argument.

3.3) YOLOv5 Post-Processing

Basically, we go through all txt files recovering the multiple classes and the bounding boxes dimensions predicted by our YOLOv5x. Then, we reshape the dimensions of each bounding boxes using the 'yolo2voc' function, because for training purpose each bounding box has been reduced in order to have values between 0 and 1. Now, we want the real dimensions of each 
bounding boxes, so we need to multiply them by the dimensions of the image (that's why we are using the dataframe coming from 'sample_submission_with_sizes.csv').

We create a new dataframe by merging our initial submission dataframe and the dataframe coming from the predictions made by YOLOv5x. Then, whenever a line is missing means that there are "No Finding" from the YOLOv5x on that particular image, so the prediction will be "14 1 0 0 1 1" corresponding to the 14th class ("No Finding") with a score of 1 and a 
bounding box at the top left of the image. Finally, we save our YOLOv5x model and our predictions as csv file.


We need to run the Notebook 'VinBigData_YOLOv5x.ipynb' five times in order to have the all 5 KFolds. Each time, we need to change the value of the fold that will generate a different 'training' and 'validation' dataframes. After saving these 5 models of YOLOv5x trained and their respective predictions, we can open and run the 'VinBigData_Ensemblist_Method.ipynb' Notebook.

4) Ensemblist Method
   
For this final Notebook, we need to import all our submission csv's (5 from YOLOv5x and 1 from ResNet18). Then, we only need to import this time the csv 'sample_submission_with_sizes.csv'. This Notebook is pretty short, because we only have to combine our different results obtained. 

We assemble the 5 different YOLOv5x submissions first, using the "WBF" method in order to obtain a unique and final prediction. For this ensemblist method, we made the choice to give the same weight value for each submission (value of 1), but it could have been interested to test different combinations and see which one give us the best Kaggle score. As a result of the "WBF" method with an IoU threshold of 0.3, we obtain a csv file 'ensemble_yolo_standard.csv'. 

The idea now is to combine the predictions made by our YOLOv5x Ensemblist Method with the ones made by our ResNet18. For doing so, we need to look each time the ResNet18 classified an image as "No Finding" and then look its related score. If this score is greater than a specific threshold (in our case 0.9) then, the prediction for this image will be "14 1 0 0 1 1". However, if the score is lower than this threshold, we add the prediction "14 1 0 0 1 1" to the one made by the YOLOv5x Ensemblist Method. In the case, where the ResNet18 classified an image as "Finding" we keep automatically the predictions made by our YOLOv5x Ensemblist Method.

From this combination, between the YOLOv5x Ensemblist Method and the ResNet18, we obtain a final submission csv as 'submission_ensemble_yolov5_resnet18.csv'. Then, we can submit this csv on Kaggle score and obtain a measure of our performance. This csv is available in the ZIP file submitted.
