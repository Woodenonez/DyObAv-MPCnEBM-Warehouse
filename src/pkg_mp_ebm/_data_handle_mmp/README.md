# Dataloader for Motion Prediction Tasks (submodule)
Contains "Dataset" and "DataHandler" for motion prediction tasks

## Data structure
The data should be structured as:
```
Dataset
|   all_data.csv
└───index1
|   |   img1
|   |   img2
|   |   ...
|   |   data.csv
└───index2
|   |   (same to index1)
└───...
```
Each index is a folder containing a CSV file and all images under this index/scene. The "data.csv" contains the index/scene data, while the "all_data.csv" contains all data extracted from each index/scene data.

## Index/scene data
An index indicates some source info of the contained data. An index can be 
- A string of the scene name, indicating where the data is collected;
- A number (int) of the batch, indicating which batch the data is collected in.
Each index folder should contain a "data.csv" CSV file and at least one image. The image is normally the background of the scene.
The CSV file records the position info of the targeted objects:

| t     | id    | index | x     | y     |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 1 | 'park' | 100 | 100 |
| 1 | 1 | 'park' | 102 | 99 |
...

If there are more than one image in the folder, it means the envrionment is **dynamic**, and each image corresponds to a time instant.

## All data
The "all_data.csv" CSV file is the integration of all index data. 

| p0    | ...   | pp    | t     | id    | index | T1    | ...   | Tf    |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1_5_0 | ...   | 1_0_p | 5     | 1     | 'park' | 1_6  | ...   | 1_16  |
...

"p_i" is the i-th past position in the form of “x_y_i”;
“T_i” is the i-th future position in the form of “x_y”.



