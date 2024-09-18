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
2) The preprocessing of these data, in order to decrease their frequency, remove some undesired activities, scale them and 
create timeseries using the **sliding-windows** method. This step is crucial because the data are given in specific timestamps 
and therefore we can't shuffle them or discover any temporal dependencies.
3) Create and compile the desired models, in order to make predictions and test the efficiency our various data.


In order to run the files, you need to download the project, go to the desired folder and read the instructions there. You should 
run each file via terminal from its folder so as to not mess with the given paths. Before running the files make sure you have 
all the necessary requirements. For example:
```azure
cd domino_dataset
python preprocess_dataset.py
```


