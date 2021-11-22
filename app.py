import streamlit as st 
import plotly.express as px
from PIL import Image

from utils import * 
from src.model import *
from config import * 

import segmentation_models_pytorch as smp 
from torchvision import transforms as T
import plotly.graph_objects as go

import cv2
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

N_CLASSES = 14

pathology_list = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


m1 = PneumothoraxModel()
m1.load_state_dict(torch.load(MODEL_DIR + 'Based-on-loss-Final_timm-efficientnet-b0_0.pt', map_location = torch.device('cpu'))['model'])
m1.eval()

m2 = PneumothoraxModel()
m2.load_state_dict(torch.load(MODEL_DIR + 'Based-on-loss-Final_timm-efficientnet-b0_1.pt', map_location = torch.device('cpu'))['model'])
m2.eval()

c1 = DenseNet121(N_CLASSES)
c1 = torch.nn.DataParallel(c1)
ckpt = torch.load(MODEL_DIR + 'model.pth.tar',map_location = torch.device('cpu'))
state_dict = ckpt['state_dict']
for key in list(state_dict.keys()):
    state_dict[key.replace('.1.', '1.'). replace('.2.', '2.')] = state_dict.pop(key)
c1.load_state_dict(state_dict)
c1.eval()

prep_fun = smp.encoders.get_preprocessing_fn(
    'timm-efficientnet-b0',
    'imagenet'
)

data_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

st.title("MedAssist")

with st.expander("Penumothorax Segmentation"):
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')

        fig = px.imshow(image)
        st.plotly_chart(fig)

        logits = get_logits(image, m1, m2, prep_fun)

        fig = px.imshow(logits, binary_string=True)
        st.plotly_chart(fig)



with st.expander("Chest X-Ray MutliLabel Classification"):
    uploaded_file = st.file_uploader("Choose a image file")

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        image_copy = image

        fig = px.imshow(image)
        st.plotly_chart(fig)

        ps, image = get_logits2(image, c1, data_transforms)

        layout = {'yaxis':  {'range': [1, 100]}}
        fig = go.Figure([go.Bar(x=PATHOLOGY_LIST, y=ps)],layout)
        st.plotly_chart(fig)

        grad_cam = GradCam(model = c1,target_layer_names='features',use_cuda = False)

        img_class = []

        label = st.selectbox('Select Class for GradCam',PATHOLOGY_LIST)

        num = PATHOLOGY_LIST.index(label)

        cam = grad_cam(image,num)
        cam = cv2.resize(cam, image_copy.size,cv2.INTER_NEAREST)
        cam = cam/np.max(cam)

        plt.imshow(image_copy)
        plt.imshow(cam, cmap='magma', alpha=0.5)
        st.pyplot()
        
