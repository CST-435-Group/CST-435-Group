"""
Mock torchvision module to prevent import errors
This prevents transformers from trying to import the actual torchvision
Also provides minimal torchvision.transforms for GAN router
"""

import sys
import types
from importlib.machinery import ModuleSpec
from PIL import Image
import torch

class OpsMock:
    """Mock ops module"""
    @staticmethod
    def nms(*args, **kwargs):
        raise NotImplementedError("NMS not available in mock torchvision")

class ToPILImageMock:
    """Mock ToPILImage transform"""
    def __init__(self, mode='RGB'):
        self.mode = mode
    
    def __call__(self, tensor):
        """Convert tensor to PIL Image"""
        if isinstance(tensor, torch.Tensor):
            # Denormalize if needed and convert to numpy
            if tensor.dim() == 3 and tensor.shape[0] == 3:
                # Shape: (C, H, W)
                tensor = tensor.permute(1, 2, 0)  # To (H, W, C)
            
            # Convert to uint8
            np_array = tensor.cpu().detach().numpy()
            if np_array.max() <= 1.0:
                np_array = (np_array * 255).astype('uint8')
            else:
                np_array = np_array.astype('uint8')
            
            return Image.fromarray(np_array, mode=self.mode)
        return tensor

class TransformsMock:
    """Mock transforms module for basic operations"""
    ToPILImage = ToPILImageMock
    
    @staticmethod
    def Compose(transforms_list):
        """Mock Compose transform"""
        class ComposeMock:
            def __init__(self, transforms):
                self.transforms = transforms
            
            def __call__(self, img):
                for t in self.transforms:
                    img = t(img)
                return img
        
        return ComposeMock(transforms_list)
    
    @staticmethod
    def ToTensor():
        """Mock ToTensor"""
        def to_tensor(x):
            return torch.from_numpy(x).float()
        return to_tensor
    
    @staticmethod
    def Normalize(mean, std, inplace=False):
        """Mock Normalize"""
        def normalize(x):
            if isinstance(x, torch.Tensor):
                mean_t = torch.tensor(mean).view(-1, 1, 1)
                std_t = torch.tensor(std).view(-1, 1, 1)
                return (x - mean_t) / std_t
            return x
        return normalize

class UtilsMock:
    """Mock utils module"""
    @staticmethod
    def save_image(tensor, fp, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0, format=None):
        """Mock save_image - converts tensor to PIL and saves"""
        to_pil = ToPILImageMock()
        if tensor.dim() == 4:
            # Batch of images - use first one
            tensor = tensor[0]
        pil_img = to_pil(tensor)
        pil_img.save(fp)
    
    @staticmethod
    def make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
        """Mock make_grid - returns the tensor as-is"""
        return tensor

# Create a proper module object for torchvision
torchvision_mock = types.ModuleType('torchvision')
torchvision_mock.__spec__ = ModuleSpec('torchvision', None)
torchvision_mock.__file__ = __file__
torchvision_mock.extension = None
torchvision_mock.ops = OpsMock()

# Create submodules
ops_mock = types.ModuleType('torchvision.ops')
ops_mock.__spec__ = ModuleSpec('torchvision.ops', None)
ops_mock.nms = OpsMock.nms

transforms_mock = types.ModuleType('torchvision.transforms')
transforms_mock.__spec__ = ModuleSpec('torchvision.transforms', None)
for attr in dir(TransformsMock):
    if not attr.startswith('_'):
        setattr(transforms_mock, attr, getattr(TransformsMock, attr))

utils_mock = types.ModuleType('torchvision.utils')
utils_mock.__spec__ = ModuleSpec('torchvision.utils', None)
for attr in dir(UtilsMock):
    if not attr.startswith('_'):
        setattr(utils_mock, attr, getattr(UtilsMock, attr))

# Replace the modules in sys.modules
sys.modules['torchvision'] = torchvision_mock
sys.modules['torchvision.ops'] = ops_mock
sys.modules['torchvision.transforms'] = transforms_mock
sys.modules['torchvision.utils'] = utils_mock
