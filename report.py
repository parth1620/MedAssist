import streamlit as st 

st.title('MedAssist Report')

st.write(
    'MedAssist is a simple healthcare diagnosis application which consist of two modeles, Chest X-Ray Multilabel Classification and Pneumothorax Segmentation.'
)

st.markdown(
    '- **Chest X-Ray Multilabel Classification**'
)

st.image('imgs/model1.png')

st.write(
    'This model takes an image of chest X-ray and outputs 14 different diseases with probabilities. Beside this it also gives a proof of decision.'
)

st.write(
    'Training is absolutely state of the art. We used weighted binary crossentropy loss which is proven to be useful for class imbalance problem. We used DenseNet121 with initialize weights of imagenet.'
)

st.image('imgs/DenseNet.jpg')

st.write(
    'Dataset that we have used is NIH Chest X-ray Dataset which is open sourced by NIH. This NIH Chest X-ray Dataset is comprised of 112,120 X-ray images with disease labels from 30,805 unique patients. To create these labels, the authors used Natural Language Processing to text-mine disease classifications from the associated radiological reports. The labels are expected to be >90% accurate and suitable for weakly-supervised learning.'
)

st.write('This dataset was having a proper validation dataset, so we did not created any validation schema.')

st.write('Lets talk about training, we used Ranger optimizer which is a combination of Lookahead + RAdam optimizer. Moreover we used CosineAnnealingLR scheduler with initial LR as 0.001, T_max as 40, Eta_min 1e-6. Finally our objective function was Weighted BCE to tackle class imbalance problem.')
st.write('The metric used for model evaluation is AUC ROC Curve.')

st.markdown('''
|     Pathology      | [Wang et al.](https://arxiv.org/abs/1705.02315) | [Yao et al.](https://arxiv.org/abs/1710.10501) | [CheXNet](https://arxiv.org/abs/1711.05225) | Our Implemented CheXNet |
| :----------------: | :--------------------------------------: | :--------------------------------------: | :--------------------------------------: | :---------------------: | 
|    Atelectasis     |                  0.716                   |                  0.772                   |                  0.8094                  |         0.8274          |
|    Cardiomegaly    |                  0.807                   |                  0.904                   |                  0.9248                  |         0.9180          | 
|      Effusion      |                  0.784                   |                  0.859                   |                  0.8638                  |         0.8770          |
|    Infiltration    |                  0.609                   |                  0.695                   |                  0.7345                  |         0.7243          | 
|        Mass        |                  0.706                   |                  0.792                   |                  0.8676                  |         0.8597          |
|       Nodule       |                  0.671                   |                  0.717                   |                  0.7802                  |         0.7871          |
|     Pneumonia      |                  0.633                   |                  0.713                   |                  0.7680                  |         0.7749          |
|    Pneumothorax    |                  0.806                   |                  0.841                   |                  0.8887                  |         0.8716          |
|   Consolidation    |                  0.708                   |                  0.788                   |                  0.7901                  |         0.8152          |
|       Edema        |                  0.835                   |                  0.882                   |                  0.8878                  |         0.8933          |
|     Emphysema      |                  0.815                   |                  0.829                   |                  0.9371                  |         0.9256          |
|      Fibrosis      |                  0.769                   |                  0.767                   |                  0.8047                  |         0.8305          |
| Pleural Thickening |                  0.708                   |                  0.765                   |                  0.8062                  |         0.7835          |
|       Hernia       |                  0.767                   |                  0.914                   |                  0.9164                  |         0.9107          |'''
)

for i in range(2):
    st.write(' ')

st.write(
    '[Lets see a preview](https://share.streamlit.io/parth1620/medassist/app.py)'
)

for i in range(3):
    st.write(' ')

st.markdown(
    '- **Pneumothorax Segmentation**'
)

st.image('imgs/__results___17_1 (2).png')

st.write(
    'This is a segmentation model which chest x-ray image as input and gives an output mask which consist penumothorax possibilities. In this we used state of the art U-Net architecture which is an AntoEncoder model.'
)

st.write(' ')

st.image('imgs/WhatsApp Image 2021-11-22 at 8.16.26 PM.jpeg')

for i in range(3):
    st.write(' ')

st.write(' Lets talk about the training, we used heavy augmentations on input images and masks at the first stage. We then passes this augmented data to Unet architecture with a efficientnet-b0 architecture. The objective or loss function used here is the addition of focal loss and dice loss. To minimize this loss we used Ranger Optimizer which is LookAhead optimizer base optimizer as RAdam. ')

st.write(' We used SIIM-ACR Pneumothorax Segmentation Dataset which is readily available on Kaggle.')

st.write('For Validation we used 5 fold cross validation. During Inference time we used 5 model as per 5 folds and output from the 5 fold model is then averaged as final output. Evaluation are as follow')

st.write(' ')
st.write(' Per Epoch Train Loss')
st.image('imgs/logs_per_epoch_train_loss.png')

st.write(' ')
st.write(' Per Epoch Valid Loss')
st.image('imgs/logs_per_epoch_valid_loss.png')

for i in range(2):
    st.write(' ')

st.write(
    '[Lets see a preview](https://share.streamlit.io/parth1620/medassist/app.py)'
)