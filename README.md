\# Machine Learning Assignment 2 – Bluetooth Antenna Fault Classification



\*\*Programme:\*\* M.Tech. Artificial Intelligence and Machine Learning (AIML), BITS Pilani WILP  

\*\*Course:\*\* Machine Learning  

\*\*Assignment:\*\* Assignment 2  

\*\*Author:\*\* Ganesh Pawar  

\*\*BITS ID:\*\* 2026AJ05935



\---



\## A. Problem Statement



The objective of this project is to develop and evaluate machine-learning classification models that identify the Bluetooth (BT) status of an antenna based on its physical and electrical characteristics.



This is a binary classification problem. For each antenna record, the model predicts one of two target classes:



\- \*\*Fault\*\* – the antenna record is classified as having a Bluetooth fault.

\- \*\*Normal\*\* – the antenna record is classified as normal.



Thirteen numerical antenna parameters are used as input features. Five classification algorithms are implemented and evaluated on the same dataset:



1\. Logistic Regression

2\. Decision Tree Classifier

3\. K-Nearest Neighbors (KNN)

4\. Gaussian Naive Bayes

5\. Support Vector Machine (SVM)



The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).



The trained models are saved as Joblib artifacts and loaded by a Streamlit application. The application accepts labeled test data in CSV format and displays individual model metrics, a confusion matrix, a classification report, and a comparison of all five models.



\*\*Scope of the Streamlit application:\*\* The application loads previously trained models and evaluates uploaded test data. It does not retrain models using the uploaded test data.



\---



\## B. Dataset Description



\### B.1 Dataset Overview



The project uses the Bluetooth antenna fault dataset read in the training notebook as `antenna\_fault.csv`.



| Dataset Attribute | Details |

|---|---|

| Dataset file used in the notebook | antenna\_fault.csv |

| Original dataset size | 2,688 records |

| Original number of columns | 17 |

| Records after duplicate removal | 1,052 |

| Input features selected for classification | 13 numerical features |

| Target variable | BT Status |

| Classification type | Binary classification |

| Target classes | Fault, Normal |



The notebook removes exact duplicate records before model development. After duplicate removal, 1,052 records remain.



\*\*Dataset Source:\*\* The training notebook identifies the input file as `antenna\_fault.csv`, but it does not record the original Kaggle or UCI dataset URL. Add the original public dataset source URL here before final submission, if applicable. The source should not be guessed or attributed without verification.



\### B.2 Input Features



The following 13 numerical features are used to train and evaluate the classifiers.



| No. | Feature |

|---|---|

| 1 | Length |

| 2 | Width |

| 3 | Height |

| 4 | Permittivity |

| 5 | Conductivity |

| 6 | Bend |

| 7 | Feed |

| 8 | S11 |

| 9 | VSWR |

| 10 | Gain |

| 11 | Efficiency |

| 12 | Bandwidth |

| 13 | epsilon\_r |



\### B.3 Target Variable and Class Distribution



The target variable is `BT Status`, with two possible labels: `Fault` and `Normal`.



After duplicate removal, the class distribution is:



| Target Class | Number of Records | Percentage |

|---|---:|---:|

| Fault | 893 | 84.89% |

| Normal | 159 | 15.11% |

| \*\*Total\*\* | \*\*1,052\*\* | \*\*100%\*\* |



The dataset is imbalanced, with the Fault class representing 84.89% of the cleaned records.



Therefore, Accuracy is considered alongside Precision, Recall, F1 Score, AUC, and MCC when comparing model performance.



\### B.4 Training and Testing Split



The notebook uses a stratified 80:20 train-test split with `random\_state=42`.



| Dataset Partition | Number of Records |

|---|---:|

| Training Set | 841 |

| Testing Set | 211 |

| \*\*Total Cleaned Records\*\* | \*\*1,052\*\* |



The stratified split preserves the target-class proportions in the training and testing partitions.



\---



\## C. GitHub Repository Link



\*\*GitHub Repository:\*\*  

https://github.com/GaneshP1111/ML-Assignment-2



The repository contains the project source code, Jupyter notebooks, saved model artifacts, dependency file, and labeled test dataset.



\### C.1 Repository Structure



The expected project structure is:



```text

ML-Assignment-2/

│

├── app.py

├── README.md

├── requirements.txt

├── test\_data.csv

│

├── model/

│   ├── logistic\_regression.joblib

│   ├── decision\_tree.joblib

│   ├── knn.joblib

│   ├── gaussian\_nb.joblib

│   └── svm.joblib

│

├── ML\_2\_Assign\_ (1).ipynb

└── ML\_Model\_Serialization\_and\_Joblib\_Generation.ipynb

