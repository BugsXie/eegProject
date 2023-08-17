# Deep Learning-Based Prediction of EEG-Based Mental Disorder Conditions

## Content
This repository includes all the content of this work, such as: 
1. In the early stage, I attempted to manually extract features and use the BiLSTM model for prediction. 

    The files involved include [models](./models), [notebooks](./notebooks), [utils](./utils), [config.py](./config.py), and [main.py](./main.py), Specific file descriptions are provided in the following text : Code file description.

2. Use the Torcheeg library package to automatically extract features and use EEGNet, GNN, and DGCNN models for prediction.

    The files involved include [torcheegProj](./torcheegProj), Specific file descriptions are provided in the following text : Code file description.

## Code file description

### [File: backup](./backup)
This is an insignificant backup folder that includes some code used for temporary testing purposes, as well as files saved from drawing models.In addition, attempts were made with respect to seeding public datasets.

### [File: data](./data)
This is a folder of datasets, which includes: 
1. Public datasets: [DEAP](./data/DEAP), [SEED](./data/SEED) 
2. Manually extracted EEG data features: [extract_data ](./data/extract_data)
3. Data quality files: [Quality](./data/Quality) 
4. Main training data [Test_EEG](./data/Test_EEG), with specific details provided in the [说明.txt](./data/Test_EEG/说明.txt).

### [File: models](./models)
This section includes the first part mentioned in the previous Content section, which pertains to the configuration file of the BiLSTM model used during the initial manual feature extraction phase.

### [File: notebooks](./notebooks)
This section encompasses the first part mentioned in the preceding Content section, which consists of a series of codes used for manual feature extraction in the preliminary stage.

### [File: torcheegProj](./torcheegProj)
This section encompasses the second part mentioned in the preceding Content section, Use the Torcheeg library package to automatically extract features and use EEGNet, GNN, and DGCNN models for prediction.

Specific Instructions:

Taking the DGCNN file as an example, it includes the main file, [DGCNN.ipynb](./torcheegProj/DGCNN/DGCNN.ipynb); the test file, [modelTest.ipynb](./torcheegProj/DGCNN/modelTest.ipynb) (used for double-blind testing); the [log](./torcheegProj/DGCNN/log) folder for recording a series of loss and accuracy changes during model training; the [models](./torcheegProj/DGCNN/models) folder containing a series of models saved during the training process; and other files for storing features and data.

### [File: utils](./utils)
This section encompasses the first part mentioned in the preceding Content section, which involves a series of tools used for manual feature extraction and prediction using the BiLSTM model in the preliminary stage. These tools include basic data preprocessing and dataset construction.

### [.gitignore](./.gitignore)
This file primarily instructs to exclude certain files, such as the configuration file for the 'data' directory, from remote synchronization with GitHub.

### [config.py](./config.py)
This section includes a series of model training parameter configuration files for the first part mentioned in the previous Content section.

### [main.py](./main.py)
This section comprises the main model training execution files for the first part mentioned in the previous Content section.

### [requirements.txt](./requirements.txt)
This section introduces the primary dependency packages utilized in this project.

## Environment
To set up the environment, you can use the pre-configured eegEnv environment to run this project.

## Contact
If you need any help, feel free to start an issue or send me an email: [bugs_xie@outlook.com](mailto:bugs_xie@outlook.com)
