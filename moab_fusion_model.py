
import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from fvcore.nn import FlopCountAnalysis
from torchinfo import summary 



class conv_(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Conv_ = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),   ### fix it by tunning [1,3,7]
            nn.Dropout(p=0.02)
            )

    def forward(self, x):
        return self.Conv_(x)
    


                #### MLP model ####  
class Linear_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Linear_ = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_channels)
            )

    def forward(self, x):
        return self.Linear_(x)
    
class MLP_Genes(nn.Module):
    def __init__(self, num_class=3):
        super(MLP_Genes, self).__init__()
        self.layer_1 = Linear_Layer(80, 80)
        self.layer_2 = Linear_Layer(80, 40)
        self.layer_3 = Linear_Layer(40, 32)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.dropout(x) 
        x = self.layer_3(x)
        return x



                #### CNN model ####                

''' One can try Vgg19_bn or Convnext, or any CNN backboane '''

model_urls = {
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
}



class PathNet(nn.Module):

    def __init__(self, features, path_dim=32, act=None, num_classes=3):
        super(PathNet, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05)
        )

        self.linear = nn.Linear(path_dim, 32)
        self.act = act



    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = self.classifier(x)
        project = self.linear(features)

        if self.act is not None:
            project = self.act(project)

        return  project



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {

    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def get_vgg(arch='vgg19_bn', cfg='E', act=None, batch_norm=True, label_dim=3, pretrained=True, progress=True):
    model = PathNet(make_layers(cfgs[cfg], batch_norm=batch_norm), act=act, num_classes=label_dim)
    
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        for key in list(pretrained_dict.keys()):
            if 'classifier' in key: pretrained_dict.pop(key)

        model.load_state_dict(pretrained_dict, strict=False)
        print("Initializing Path Weights")

    return model

            #####  ConvNeXt model #####
        
from torchvision import models

class convNext(nn.Module):
    def __init__(self, n_classes=32):
        super().__init__()
        convNext = models.convnext_base(pretrained=True)
        convNext.avgpool = nn.AdaptiveAvgPool2d((1))
        convNext.classifier = nn.Sequential(nn.Flatten(1, -1),
                                            nn.Dropout(p=0.2),
                                            nn.Linear(in_features=1024, out_features=n_classes)
                                            )
        self.base_model = convNext


    def forward(self, x):
        x = self.base_model(x)
        return x


             ### Outer subtraction ###
             

def append_0_s(x1,x2): 
    b = torch.tensor([[0]]).to(device=DEVICE,dtype=torch.float32)
    x1 = torch.cat((b.expand((x1.shape[0],1)),x1),dim=1)
    #print('this is x1 and this is the shape of x1',x1.shape)
    
    x2 = torch.cat((b.expand((x2.shape[0],1)),x2),dim=1)
    #print('this is x1 and this is the shape of x1',x2.shape)

    x_p = x2.view(x2.shape[0], x2.shape[1], 1) - x1.view(x1.shape[0], 1, x1.shape[1])
    x_p = torch.sigmoid(x_p)
    #print('the shape of xp after outer add bfr flatten',x_p.shape)
    #x_p = x_p.flatten(start_dim=1)
    return x_p

                ### Outer addition ###

def append_0(x1,x2): 
    b = torch.tensor([[0]]).to(device=DEVICE,dtype=torch.float32)
    x1 = torch.cat((b.expand((x1.shape[0],1)),x1),dim=1)
    #print('this is x1 in add and this is the shape of x1',x1.shape)
    
    x2 = torch.cat((b.expand((x2.shape[0],1)),x2),dim=1)
    #print('this is x1 and this is the shape of x1',x2.shape)

    x_p = x2.view(x2.shape[0], x2.shape[1], 1)+ x1.view(x1.shape[0], 1, x1.shape[1])
    x_p = torch.sigmoid(x_p)
    #print('the shape of xp after outer add bfr flatten',x_p.shape)
    #x_p = x_p.flatten(start_dim=1)
    return x_p
    

                ### Outer product ###

def append_1(x1,x2):
    b = torch.tensor([[1]]).to(device=DEVICE,dtype=torch.float32)
    x1 = torch.cat((b.expand((x1.shape[0],1)),x1),dim=1)
    #print('this is x1 of OP and this is the shape of x1',x1.shape)
    
    x2 = torch.cat((b.expand((x2.shape[0],1)),x2),dim=1)
    #print('this is x1 and this is the shape of x1',x2.shape)

    x_p = x2.view(x2.shape[0], x2.shape[1], 1)* x1.view(x1.shape[0], 1, x1.shape[1])
    x_p = torch.sigmoid(x_p)
    #print('the shape of xp after outer pro bfr flatten',x_p.shape)
    #x_p = x_p.flatten(start_dim=1)
    return x_p

                ### Outer division ###

def append_1_d(x1,x2):
    b = torch.tensor([[1]]).to(device=DEVICE,dtype=torch.float32)
    x1 = torch.cat((b.expand((x1.shape[0],1)),x1),dim=1)
    #print('this is x1 of div and this is the shape of x1',x1.shape)
    
    x2 = torch.cat((b.expand((x2.shape[0],1)),x2),dim=1)
    
    x1_ = torch.full_like(x1, fill_value=float(1.2e-20))  #this to avoid division by zeor, in this case x1 is the denominator 
    x1 = torch.add(x1, x1_)
    
    x_p = x2.view(x2.shape[0], x2.shape[1], 1)/ x1.view(x1.shape[0], 1, x1.shape[1])
    x_p = torch.sigmoid(x_p)
    #print('the shape of xp after outer pro bfr flatten',x_p.shape)
    #x_p = x_p.flatten(start_dim=1)
    return x_p

    
                #### Fusion model ####
                

class MOAB(nn.Module):
    def __init__(self, model_image,model_gens,nb_classes=3):
        super(MOAB, self).__init__()
        self.model_image =  model_image
        self.model_gens = model_gens       
        self.fc = nn.Linear(1089, 512) # the shape of the flttened x after using conv_stack is (33x33 = 1089)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_out = nn.Linear(512, nb_classes)
        
        self.conv_stack= conv_(4,1)
    
    def forward(self, x1,x2):
        
        #The shape of the image (x1) in this case has already been flattened by the context pre-trained network. 
        x1 = self.model_image(x1)
        
        x2 = self.model_gens(x2)

        # This is done to flatten the feature map from the MLP layer.
        x2 = x2.view(x2.size(0), -1)

        # The objective of adding an extra dim to each branch (for example, torch.unsqueeze(x_sub, 1)) is to assist us in combining along the channel dim, so the shape of x_sub would be (bs, channel,33,33) 
               
        ## outer addition branch (appending 0)
        x_add = append_0(x1,x2)
        x_add = torch.unsqueeze(x_add, 1)
        ## outer subtraction branch (appending 0)
        x_sub = append_0_s(x1,x2)
        x_sub = torch.unsqueeze(x_sub, 1)
        #print('out add shape', x_add.shape)
        #print('batch size add shape', x_add.shape[0])

        ## outer product branch (appending 1)
        x_pro =append_1(x1,x2)
        x_pro = torch.unsqueeze(x_pro, 1)
        
        
        ## outer divison branch (appending 1)
        x_div =append_1_d(x1,x2)
        x_div = torch.unsqueeze(x_div, 1)
        
        ## combine 4 branches on the channel dim
        x = torch.cat((x_add,x_sub,x_pro,x_div),dim=1)
        #print('shape afr cat', x.shape)
        
        ## use a conv (1x1) 
        x = self.conv_stack(x)
        #print('shape after conv', x.shape)
        x = x.flatten(start_dim=1)
        
        #print('shape aftr flatten', x.shape)
        
        x = self.fc(x)
        #print('fc after combined', x.shape)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x
    
#%%
img = convNext()
mlp = MLP_Genes()
model = MOAB(img,mlp)  
model = model.to(device=DEVICE,dtype=torch.float)
print(summary(model,[(8,3, 224, 224),(8,80)]))
#%%


# Count total parameters
total_params = sum(p.numel() for p in model.parameters())

# Count only trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params}")
print(f"MOAB Trainable Parameters: {trainable_params}")
#%%
# Count FLOPs
img_input =  torch.randn(1,3, 224, 224)
omic_input  = torch.randn(1,80)
flops = FlopCountAnalysis(model, (img_input,omic_input))
print(f"\nTotal FLOPs: {flops.total():,}")
print(f"Total GFLOPs: {flops.total() / 1e9:.6f}")

