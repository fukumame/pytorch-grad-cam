import torch
from torch.autograd import Variable
from torch.autograd import Function, NestedIOFunction
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms
from pathlib import Path


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


class GradCam:
    def __init__(self, model, target_layer_names, device):
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.device = device

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        input.to(self.device)
        features, output = self.extractor(input)

        input_w = input.shape[3]
        input_h = input.shape[2]

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.Tensor(one_hot)
        one_hot.requires_grad = True
        one_hot.to(self.device)
        one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_w, input_h))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, device):
        self.model = model
        self.model.eval()
        self.device = device
        self.model = model.to(self.device)

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        input.to(self.device)
        output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.Tensor(one_hot)
        one_hot.requires_grad = True
        one_hot.to(self.device)
        one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output

class ImageProcessor:

    @staticmethod
    def preprocess_image(img, height, width):
        transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        transformed_img = transform(img)
        transformed_img = torch.unsqueeze(transformed_img, 0)
        transformed_img.requires_grad = True

        return transformed_img

    @staticmethod
    def save_cam_on_image(img, mask, file_path, heatmap_ratio):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap * heatmap_ratio + np.float32(img) * (1 - heatmap_ratio)
        cam = cam / np.max(cam)
        cv2.imwrite(str(file_path), np.uint8(255 * cam))

    @staticmethod
    def convert_image_to_array(img):
        image_array = np.array(img, dtype=np.uint8)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        image_array = image_array /255
        return image_array
    @staticmethod
    def output_image_name(image_path, type, output_dir):
        p = Path(image_path)
        file_name = p.stem
        return Path(output_dir) / f"{file_name}_{type}.jpg"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', "-i",type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--output_dir', '-o', type=str, default='result', help='Directory path to output result')
    parser.add_argument('--img_height', '-ih', type=int, default=224, help='image height of model input')
    parser.add_argument('--img_width', '-iw', type=int, default=224, help='image width of model input')
    parser.add_argument('--heatmap_ratio', '-hr', type=float, default=0.5, help='mixture ratio of heatmap')
    parser.add_argument('--output_gb', '-g', action="store_true", help='mixture ratio of heatmap')
    parser.add_argument('--output_cam_gb', '-cg', action="store_true")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    device = get_device()

    grad_cam = GradCam(model=models.vgg19(pretrained=True),
                       target_layer_names=["35"], device=device)
    img = Image.open(args.image_path)
    img = img.resize((args.img_width, args.img_height))
    input_img = ImageProcessor.preprocess_image(img, height=args.img_height, width=args.img_width)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None

    mask = grad_cam(input_img, target_index)
    image_array = ImageProcessor.convert_image_to_array(img)

    cam_file_path = ImageProcessor.output_image_name(args.image_path, "cam", Path(args.output_dir))

    ImageProcessor.save_cam_on_image(image_array, mask, cam_file_path, args.heatmap_ratio)

    gb_model = GuidedBackpropReLUModel(model=models.vgg19(pretrained=True), device=device)
    gb = gb_model(input_img, index=target_index)
    if args.output_gb:
        gb_file_path = ImageProcessor.output_image_name(args.image_path, "gb", Path(args.output_dir))
        utils.save_image(torch.from_numpy(gb), gb_file_path)

    if args.output_cam_gb:
        cam_mask = np.zeros(gb.shape)
        for i in range(0, gb.shape[0]):
            cam_mask[i, :, :] = mask
        cam_gb = np.multiply(cam_mask, gb)
        can_gb_file_path = ImageProcessor.output_image_name(args.image_path, "cam_gb", Path(args.output_dir))
        utils.save_image(torch.from_numpy(cam_gb), can_gb_file_path)
