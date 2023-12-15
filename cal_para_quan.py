from model import VGG,AlexNet,ResNet50,Classifier,Encoder,Decoder,GreenBlock,AE

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Instantiate the model
model1 = Encoder()

# Calculate the number of parameters
total_params = count_parameters(model1)
print(f"Total number of parameters: {total_params}")
