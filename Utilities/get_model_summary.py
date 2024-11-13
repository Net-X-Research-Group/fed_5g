from torchinfo import summary

from task import Net

model = Net()
batch_size = 16
depth = 3
height, width = 32, 32
summary(model, input_size=(batch_size, depth, height, width), device='cpu', col_names=["kernel_size", "output_size", "num_params"],
    row_settings=["var_names"])
print(model)