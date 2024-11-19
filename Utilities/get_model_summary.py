from torchinfo import summary

from task import Net
from models import EmbeddedNet

model = Net()
batch_size = 64
depth = 3
height, width = 32, 32
summary(model, input_size=(16,3,32,32), device='cpu', col_names=["kernel_size", "output_size", "num_params"],
    row_settings=["var_names"])
print(model)