from fastapi import FastAPI, File, UploadFile

from io import BytesIO

from PIL import Image

# import torch

# import torch.nn as nn

# import torchvision.transforms as transforms

# import timm

app=FastAPI(title='predict cassava leaf disease')

# valid_transforms=transforms.Compose([transforms.Resize((224,224)),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# class ViTBase16(nn.Module):
#     def __init__(self, n_classes, pretrained):
#         super().__init__()
#         self.model=timm.create_model('vit_base_patch16_224', pretrained=pretrained)
#         self.model.head=nn.Linear(self.model.head.in_features, n_classes)
#     def forward(self,x):
#         x=self.model(x.to('cpu'))
#         return x
#     def train_one_epoch(self, train_loader, criterion, optimizer, device):
#         epoch_loss=0
#         epoch_accuracy=0
#         self.model.train()
#         for image, label in train_loader:
#             image,label=image.cpu(), label.cpu()
#             optimizer.zero_grad()
#             output=self.forward(image)
#             loss=criterion(output, label)
#             loss.backward()
#             accuracy=(output.argmax(dim=1)==label).float().mean()
#             epoch_loss+=loss
#             epoch_accuracy+=accuracy
#             optimizer.step()

#         return epoch_loss/len(train_loader), epoch_accuracy/len(train_loader)
#     def validate_one_epoch(self, valid_loader, criterion, device):
#         valid_loss=0
#         valid_accuracy=0
#         self.model.eval()
#         for image, label in valid_loader:
            
#             image, label=image.cpu(), label.cpu()
#             with torch.no_grad():
#                 output=self.model(image)
#                 loss=criterion(output, label)
#                 accuracy=(output.argmax(dim=1)==label).float().mean()
#                 valid_loss+=loss
#                 valid_accuracy+=accuracy
#         return valid_loss/len(valid_loader), valid_accuracy/len(valid_loader)
            

# model=ViTBase16(n_classes=5, pretrained=False)

# model.load_state_dict(torch.load('model_weights.pth'))


# import json

# with open('label_num_to_disease_map.json') as f:
#     f=json.load(f)

@app.post('/predict')
async def predict(file: UploadFile= File(...)):
    img=Image.open(BytesIO(await file.read())).convert('RGB')
    # img=valid_transforms(img)
    # pred_idx=torch.argmax(model(torch.unsqueeze(img,0)),1)
    
    # return f[str(pred_idx.item())]
    return 'test'