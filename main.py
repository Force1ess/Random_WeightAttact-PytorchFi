'''
Author: Forceless
Date: 2021-10-21 14:51:42
LastEditTime: 2021-10-21 21:44:22
'''
import torch as t
import torchvision as tv
from PIL import Image
from torchvision import transforms
import os

from torchvision.models.resnet import resnet50
import pytorchfi.core as ficore
import random
from random import randint
from binary_converter import bit2float, float2bit
from tqdm import tqdm
inshape = [3, 224, 224]
vgg_16 = tv.models.vgg16(pretrained=True, progress=True)
path = "./datasets/Chosen_Images/"
image_collection = os.listdir(path)
cuda = t.cuda.is_available()
def model_process(model):
    Conv_Struc = []
    for i in model.parameters():
        if(len(i.shape) == 4):
            Conv_Struc.append([i.shape[1], i.shape[0], i.shape[2]])
    return Conv_Struc
# input_channel,output_channel,kernelsize
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
preprocess = tv.transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
def data_process(image_collection,batch):
        image_data = [preprocess(Image.open(f"./datasets/Chosen_Images/{file}")) for file in image_collection]
        data=[]
        for i in range(0,len(image_collection),batch):
                temp = t.zeros((batch,3,224,224))
                for j in range(batch):
                        temp[j]=image_data[i+j]
                if cuda:
                        temp=temp.cuda()
                data.append(temp)
        return data
def get_result(model,data,batch):
        output=[]
        for i in data:
                output.append(model(i))
        return unwrap(output,batch)
def unwrap(data,batch):
        result = []
        for i in data:
                for j in range(batch):
                        probabilities = t.nn.functional.softmax(i[j], dim=0)
                        top5_prop, top5_catid = t.topk(probabilities, 5)
                        result.append((categories[top5_catid[0]], top5_prop[0].item()))
        return result
# [layer_num,dim_in,dim_out,dim_kernel1,dim_kernel2,bit_position,result,propotion]
    
#Store
def Attack(model=vgg_16,batch=80,path=path,bitlen=32,perlayer_error_num=3000):
    image_collection = os.listdir(path)
    cuda = t.cuda.is_available()
    model.eval()
    if cuda:
        model.cuda()
    print(f"Cuda availablity : {cuda}")
    data = data_process(image_collection,batch)
    layer_struc = model_process(model)
    layer_num=len(layer_struc)
    pfimodel = ficore.fault_injection(model = model,input_shape= inshape, 
    layer_types=[t.nn.Conv2d],use_cuda=cuda,batch_size=batch)
    error_result = []
    # Only apply to conv now
    for i in range(layer_num):
        print(f"Layer {i} of {layer_num}Layers")
        for j in tqdm(range(perlayer_error_num)):
            # choose a random weight
            dim_in, dim_out, dim_kernel1, dim_kernel2 = randint(0,layer_struc[i][0]-1), randint(0, layer_struc[i][1]-1),randint(0, layer_struc[i][2]-1), randint(0, layer_struc[i][2]-1)
            # choose a random bit
            bit = random.randint(0, bitlen-1)
            def bit_flip(data, location):
                old = t.Tensor([data[location]])
                bits = float2bit(old)
                print(f"Attack : {bit}")
                if bits[0][bit] == 1:
                    bits[0][bit] = 0
                else:
                    bits[0][bit] = 1
                newData = bit2float(bits)
                if cuda:
                    newData=newData.cuda()
                return newData
            inj = pfimodel.declare_weight_fi(layer_num=i, k=dim_out, dim1=dim_in,
                                    dim2=dim_kernel1, dim3=dim_kernel2,function=bit_flip)
            if cuda:
                inj = inj.cuda()
            # 3000 error per layer on 80 images
            output = get_result(inj,data,batch)
            error_result.append([i,dim_in,dim_out,dim_kernel1,dim_kernel2,bit,output])
    t.save(error_result,'error.pt')
Attack(model=resnet50(),perlayer_error_num=1)

