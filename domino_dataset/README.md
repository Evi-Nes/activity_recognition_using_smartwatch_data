# DOMINO Dataset

The dataset can be found [here](https://dataverse.unimi.it/dataset.xhtml?persistentId=doi:10.13130/RD_UNIMI/QECFKA).

After you download the **DOMINO.tar.gz** file, make sure you place it inside this folder and extract it. The data are 
organized in separate folders by user_id, so it's necessary to combine the data with the correct labels and create a csv 
file to contains all the useful information.  

To do that, run the file **preprocess_dataset.py** file found in this folder. The data are now stored in the **data.csv** file and
are ready to be used in the project. 

The other files contain the implemented models, each file processing a different type of data, so make sure you run the 
desired one.

