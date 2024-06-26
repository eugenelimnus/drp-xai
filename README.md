# drp-xai
XAI on cancer drug response

This git repository contains the code used to implement XAI techniques on the deep learning model velodrome (https://github.com/hosseinshn/Velodrome).

In order to replicate the code, the following steps are needed:

1) Clone the Velodrome repository (https://github.com/hosseinshn/Velodrome) 
2) Follow the steps given in the Velodrome README to initialise a Velodrome model.
3) Copy the files in XAI over to the folder "Velodrome Train", note that LoadData.py should replace the old LoadData.py in folder "Velodrome Train".
4) The following files can then be run in "Velodrome Train": L2X.py, LIME.py, SHAP.py and XAI_evaluation.py

# Citation 
```
@article{sharifi2021out,
  title={Out-of-distribution generalization from labelled and unlabelled gene expression data for drug response prediction},
  author={Sharifi-Noghabi, Hossein and Harjandi, Parsa Alamzadeh and Zolotareva, Olga and Collins, Colin C and Ester, Martin},
  journal={Nature Machine Intelligence},
  pages={1--11},
  year={2021},
  publisher={Nature Publishing Group}
}
```