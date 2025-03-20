import torch
import PyTorchEncoding.encoding as encoding

# Get the model
model = encoding.models.get_model('deeplab_v3b_plus_wideresnet_citys', pretrained=True).cuda()
model.eval()

# Print layers to verify
for name, param in model.named_parameters():
    print(name, param.shape)