# Activiity Recognition Using Smartwatch Data

This is part of my diploma thesis, which I did as part of my studies as an 
Electrical and Computer Engineer at Aristotle University of Thessaloniki and can be found [here](https://ikee.lib.auth.gr/record/356521/?ln=en).

In this thesis, I explore the use of neural networks, as well as other classifiers, in human activity
recognition using data from smartwatches, in order to contribute to the improvement of the overall care for pancreatic 
cancer patients, for the [RELEVIUM](https://www.releviumproject.eu/) project of the Information Technologies Institute.


The tasks implemented in this project are:
1) Τhe accumulation of a complete dataset, covering the requirements of the problem to be solved. In our case, it was 
necessary to use data from smartwatches, describing simple physical activities of daily life. Therefore, the used datasets 
are the **DOMINO** dataset, the **PAMAP2** and the **WISDM**. 
2) The preprocessing of these data, in order to decrease their frequency, remove some undesired activities and 
create timeseries using the **sliding-windows** method. This step is crucial because the data are given in specific timestamps 
and therefore we can't shuffle them or discover any temporal dependencies.
3) Create and compile the desired models, in order to make predictions and test the efficiency our various data.


In order to run the files, you need to download the project, go to the desired folder and read the instructions there. You should 
run each file via terminal from its folder so as to not mess with the given paths. For example:
```azure
cd activity_recognition_using_smartwatch_data/domino_dataset/

python preprocess_dataset.py
```

Before running the files make sure you have all the necessary requirements, which can be found in the **requirements.txt** file.

This is what your processed data should look like. Using the sliding windows method, with overlapping, we split them to 
smaller timeseries, as shown below.
![screenshot](media/sliding_windows.png)

I implemented three types of models depending on the type of data I use each time. I have models using only accelerometer data,
models using only gyroscope data and models using both types of data. Below I show the train and test results, comparing the 
performances of the models for each type of data. It's clear that the accelerometer data are the ones with the highest results
on the test phase of our analysis and there are the ones I recommend using. The combined data have also high results and should
be considered a good alternative.

### **Train results:**
![screnshot](media/train_acc.png)

### **Test results:**
![screnshot](media/test_acc.png)

### Chosen Models
In the image below are shown the different metrics as they result from our analysis for the accelerometer data. This helps us choose the models which are the most
suitable for our needs and capture the best the patterns of our data. Those are the **GRU-2, CNN-LSTM, CNN-GRU, CNN-CNN-
LSTM**, and **CNN-CNN-GRU**. 

![screnshot](media/chart_acc.png)

### Frequency test
Furthermore, I performed some test concerning the frequency of the data. The initial frequency was 100Hz but for the needs
of our problem I had to decrease it to 25Hz. However, it is necessary to understand
the importance of the frequency of the data and its impact on the results.

![screnshot](media/freaquency_test.png)

