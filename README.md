# MVAT_Loc

MVAT_Loc：A novel Deep Learning Model for Multi-label Protein Subcellular Localization Prediction Based on Attention Mechanism with Multi-View Fusion


###Guiding principles:

You should:

**Download Dataset

       Download the image data from the ./code/0_data folder.

**Image unmix

      This part is responsible for image channel unmix. The code is in the  ./code/1_Image unmix folder. The main function is SepeProtein.m  

** Feature extraction

    *LBP Features
         This part is responsible for image feature extraction. The code is in the  ./code/2_features/LBP Features folder. The main function is LBP_ALL.m  

    *Pre_train deep model Features
         This part is responsible for image feature extraction. The code is in the  ./code/2_features/Pre_train deep model Features folder. 
         The main function is deep_feature128.py and deep_feature512.py

**Multi-label Classification model based on Multi-view fusion

        This part is responsible for Multi-label Classification. The code is in the  ./code/3_model folder. The main function is main_train.py
    
