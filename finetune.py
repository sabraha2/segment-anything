from datasets import load_dataset
from transformers import SamProcessor
from torch.utils.data import DataLoader
from segdataset import SegmentationDataset
from torchvision import transforms
from transformers import SamModel
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
import sys, os
import warnings
warnings.filterwarnings('ignore')
from torch.nn.functional import threshold, normalize

# Data and model directories
train_data_dir = "/project01/cvrl/jhuang24/australia-backup/data/test/"
save_model_path = "/project01/cvrl/jhuang24/sam_data/finetune"

# Define customized dataset
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
data_transforms = transforms.Compose([transforms.ToTensor()])

train_dataset = SegmentationDataset(root=train_data_dir,
                                    processor=processor,
                                    image_folder='imgs',
                                    mask_folder='masks',
                                    transforms=data_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

batch = next(iter(train_dataloader))

for k,v in batch.items():
  print(k,v.shape)

print("GT mask: ", batch["ground_truth_mask"].shape)


# Load pretrain model
model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

# Note: Hyperparameter tuning could improve performance here
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()

for epoch in range(num_epochs):
    epoch_losses = []

    # i = 0

    for batch in tqdm(train_dataloader):

        # forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)

        # TODO (JH): I think SAM actually used MSE loss, but this is TBD.
        loss = seg_loss(predicted_masks, ground_truth_masks)

        # backward pass (compute gradients of parameters w.r.t. loss)
        optimizer.zero_grad()
        loss.backward()

        # optimize
        optimizer.step()
        epoch_losses.append(loss.item())

    # Save model for each epoch
    # model_name = "finetune_sam_epoch_" + str(epoch).zfill(4) + ".pth"
    model.save_pretrained(save_model_path)
    # print("Model saved.")