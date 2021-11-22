import numpy as np 
import torch
import albumentations as A


def get_logits(image, m1, m2, prep_fun):

    with torch.no_grad():
        image = np.array(image)
        image = prep_fun(image)
        h, w, c = image.shape
        image = A.Resize(384, 384)(image = image)['image']
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1).unsqueeze(dim = 0)

        logits = (m1(image) + m2(image)) / 2
        logits = (torch.sigmoid(logits) > 0.5)*1.0
        logits = logits.squeeze().detach().numpy()
        logits = A.Resize(h, w)(image = logits)['image']

        return logits


def get_logits2(image, c1, data_transforms):
    with torch.no_grad():
        image = data_transforms(image)
        image = image.unsqueeze(0)
        ps = c1(image)
        ps = ps.data.numpy()
        ps = 100*ps
        ps = ps.tolist()[0]
        return ps, image


class FeatureExtractor():

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model.module.densenet121._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
                x = x.mean([2,3])
        
        return outputs, x

class ModelOutputs():

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
        
            target_activations, x = self.feature_extractor(x)
        
        return target_activations, x


class GradCam:

    def __init__(self,model,target_layer_names,use_cuda = False):

        self.model = model
        self.model.eval()
        self.cuda = use_cuda

        self.extractor = ModelOutputs(self.model,target_layer_names)

    def forward(self,input_img):
        return self.model(input_img)

    def __call__(self,input_img,index = None):

        if self.cuda:
            features, output = self.extractor(input_img.cuda())
        else:
            features, output = self.extractor(input_img)

        ps = output[0][index].requires_grad_(True)
        
        self.model.zero_grad()
        ps.backward(retain_graph = True)

        
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0,:]

        weights = np.mean(grads_val, axis = (2,3))[0,:]
        cam = np.zeros(target.shape[1:], dtype = np.float32)

        for i,w in enumerate(weights):
            cam += w * target[i, : ,:]
        
        cam = np.maximum(cam,0)
        
        return cam