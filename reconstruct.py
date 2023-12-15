import torch
from model import AE
from torchvision import transforms
import torch.nn as nn
from load_data import PretrainDataset
from PIL import Image

dataset = PretrainDataset(root_dir='./Data')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ae = AE()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    ae = nn.DataParallel(ae)
ae.to(device)

# Load the pre-trained model
model_path = './best_ae_model_all.pth'
state_dict = torch.load(model_path, map_location=device)
ae.load_state_dict(state_dict)
print("Model loaded successfully.")


images = [dataset[i] for i in range(400) if i % 8 == 0]
# Process each image and save
to_pil = transforms.ToPILImage()

for i, (augmented_image, image_tensor, norm_image) in enumerate(images):
    # Put the normalized image into the model
    norm_image = norm_image.unsqueeze(0).to(device)
    with torch.no_grad():
        ae.eval()
        output = ae(norm_image)
        output = output.squeeze(0)  # Remove batch dimension

    # Convert tensor to PIL image
    output_pil = to_pil(output.cpu())

    # Ensure augmented_image and image_tensor are also PIL images
    if not isinstance(augmented_image, Image.Image):
        augmented_image = to_pil(augmented_image)
    if not isinstance(image_tensor, Image.Image):
        image_tensor = to_pil(image_tensor.cpu())

    # Horizontally concatenate images
    combined_width = augmented_image.width + image_tensor.width + output_pil.width
    combined = Image.new('RGB', (combined_width, augmented_image.height))
    combined.paste(augmented_image, (0, 0))
    combined.paste(image_tensor, (augmented_image.width, 0))
    combined.paste(output_pil, (augmented_image.width + image_tensor.width, 0))

    # Save the combined image
    combined.save(f'./generate/combined_{i}.png')