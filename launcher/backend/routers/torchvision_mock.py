"""
Mock torchvision module to prevent import errors
This prevents transformers from trying to import the actual torchvision
"""

import sys
import types
from importlib.machinery import ModuleSpec

class OpsMock:
    """Mock ops module"""
    @staticmethod
    def nms(*args, **kwargs):
        raise NotImplementedError("NMS not available in mock torchvision")

# Create a proper module object for torchvision
torchvision_mock = types.ModuleType('torchvision')
torchvision_mock.__spec__ = ModuleSpec('torchvision', None)
torchvision_mock.__file__ = __file__
torchvision_mock.extension = None
torchvision_mock.ops = OpsMock()

# Create ops submodule
ops_mock = types.ModuleType('torchvision.ops')
ops_mock.__spec__ = ModuleSpec('torchvision.ops', None)
ops_mock.nms = OpsMock.nms

# Replace the modules in sys.modules
sys.modules['torchvision'] = torchvision_mock
sys.modules['torchvision.ops'] = ops_mock
