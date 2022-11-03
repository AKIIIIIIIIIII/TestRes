import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

def save_training_images(image, dest_folder, suffix_filename:str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    save_image(image, os.path.join(dest_folder, f"{suffix_filename}.png"))

class ColorShift():
    def __init__(self, device: torch.device='cpu', mode='uniform', image_format='rgb'):
        self.dist: torch.distributions = None
        self.dist_param1: torch.Tensor = None
        self.dist_param2: torch.Tensor = None

        if(mode == 'uniform'):
            self.dist_param1 = torch.tensor((0.199, 0.487, 0.014), device=device)
            self.dist_param2 = torch.tensor((0.399, 0.687, 0.214), device=device)
            if(image_format == 'bgr'):
                self.dist_param1 = torch.permute(self.dist_param1, (2, 1, 0))
                self.dist_param2 = torch.permute(self.dist_param2, (2, 1, 0))

            self.dist = torch.distributions.Uniform(low=self.dist_param1, high=self.dist_param2)
            
        elif(mode == 'normal'):
            self.dist_param1 = torch.tensor((0.299, 0.587, 0.114), device=device)
            self.dist_param2 = torch.tensor((0.1, 0.1, 0.1), device=device)
            if(image_format == 'bgr'):
                self.dist_param1 = torch.permute(self.dist_param1, (2, 1, 0))
                self.dist_param2 = torch.permute(self.dist_param2, (2, 1, 0))

            self.dist = torch.distributions.Normal(loc=self.dist_param1, scale=self.dist_param2)
        
    #Allow taking mutiple images batches as input
    #So we can do: gray_fake, gray_cartoon = ColorShift(output, input_cartoon)
    def process(self, *image_batches: torch.Tensor):
        # Sample the random color shift coefficients
        weights = self.dist.sample()

        # images * weights[None, :, None, None] => Apply weights to r,g,b channels of each images
        # torch.sum(, dim=1) => Sum along the channels so (B, 3, H, W) become (B, H, W)
        # .unsqueeze(1) => add back the channel so (B, H, W) become (B, 1, H, W)
        # .repeat(1, 3, 1, 1) => (B, 1, H, W) become (B, 3, H, W) again
        return ((((torch.sum(images * weights[None, :, None, None], dim= 1)) / weights.sum()).unsqueeze(1)).repeat(1, 3, 1, 1) for images in image_batches)
            

if __name__ == "__main__":
    image = "/content/TestRes/datasets/demo_edges2handbags/testA/0.jpg"
    transform = transforms.Compose([transforms.Resize(256),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Variable(transform(Image.open(image).convert('RGB')).unsqueeze(0).cuda())
    color_shift = ColorShift()
   # input1 = torch.randn(5,3,256,256)
  #  input2 = torch.randn(5,3,256,256)
    result1, result2 = color_shift.process(image,image)
    save_training_images(torch.cat((result1*0.5+0.5, result2*0.5+0.5),axis=2), "./testtest", "mono")
    print(result1.shape, result2.shape) #torch.Size([5, 3, 256, 256]) torch.Size([5, 3, 256, 256])