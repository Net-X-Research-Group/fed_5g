from torchinfo import summary

from federated_application.models import ModelWrapper

model = ModelWrapper.create_model('mobilenet_v3_small')

batch_size = 64
depth = 3
height, width = 32, 32
summary(model, input_size=(16,3,32,32), device='cpu', col_names=["kernel_size", "output_size", "num_params"],
    row_settings=["var_names"])
print(model)