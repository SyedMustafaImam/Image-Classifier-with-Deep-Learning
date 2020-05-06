import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
import sys

def args_paser():
    pa = argparse.ArgumentParser(description='predictor')
    pa.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='checkpoint to str')
    pa.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    pa.add_argument('--top_k', type=int, default=5, help='top classes')
    pa.add_argument('--img', type=str, required='True',help='Path of image')

    args = pa.parse_args()
    return args

def load_checkpoint(check_path):
    checkpoint = torch.load(check_path)
    model = getattr(models, checkpoint['structure'])(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False


    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dic'])

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    w, h = image.size

    if w > h:
        r = float(w) / float(h)
        size = 256 * r, 256
    else:
        r = float(h) / float(w)
        size = 256, 256 * r

    image.thumbnail(size, Image.ANTIALIAS)

    image = image.crop((256 // 2 - 112, 256 // 2 - 112, 256 // 2 + 112, 256 // 2 + 112))

    img_array = np.array(image)
    np_img = img_array / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    npimage = (np_img - mean) / std
    npimage = npimage.transpose(2, 0, 1)

    return npimage


def predict(image_path, model, device, cat_to_name, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file

    model.to(device)
    model.eval()

    torch_img = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to(device)
    output = torch.exp(model.forward(torch_img))
    probs, classes = output.topk(topk)

    probs = Variable(probs).cpu().numpy()[0]
    probs = [x for x in probs]

    classes = Variable(classes).cpu().numpy()[0]
    classes = [c for c in classes]
    idx_to_classes = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_classes[i] for i in classes]
    labels = [cat_to_name[l] for l in top_classes]

    return probs, top_classes, labels


def main():
    args = args_paser()
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    is_gpu = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    if is_gpu and use_cuda:
        device = torch.device("cuda:0")
        print(f"Device is set to {device}")

    else:
        device = torch.device("cpu")
        print(f"Device is set to {device}")
        
    model = load_checkpoint(args.checkpoint)
    np_image = process_image(args.img)
    topk_pro, topk_cl, topk_la= predict(args.img, model, device, cat_to_name, args.top_k)
    
    print('Predicted top classes : ', topk_cl)
    print('Flowers: ', topk_la)
    print('Probablity: ', topk_pro)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)