import json
import pickle
import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import SamModel, SamProcessor
from sklearn.metrics import f1_score, roc_auc_score, jaccard_score, roc_curve, auc
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import warnings
from segdataset import SegmentationDataset
from safetensors.torch import load_file as load_safetensor
from safetensors import safe_open
warnings.filterwarnings('ignore')

# Data and model directories
data_dir = "/project01/cvrl/jhuang24/australia-backup/data"
save_model_path = "/project01/cvrl/sabraha2/sam_data/finetune_v3_epochs_50/model.safetensors"

batch_size = 1
threshold_value = 0.5

def calculate_eer(fpr, tpr, thresholds):
    fnr = 1 - tpr
    eer_threshold_index = np.nanargmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_threshold_index]
    eer = fpr[eer_threshold_index]
    return eer, eer_threshold, eer_threshold_index

def test_model(data_directory, model_path, batch_size, threshold_value):
    # Load pre-trained model
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    
    # Ensure model is loaded correctly
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"The model file at {model_path} was not found.")
    
    tensors = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    
    model.load_state_dict(tensors)
    
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define customized dataset
    data_transforms = transforms.Compose([transforms.ToTensor()])

    dataset = SegmentationDataset(root=data_directory,
                                  processor=processor,
                                  image_folder='imgs',
                                  mask_folder='masks',
                                  transforms=data_transforms)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True)

    correct = 0
    total = 0
    jaccard = 0
    nb_sample = 0

    gt = []
    pred = []

    for batch in tqdm(dataloader):
        with torch.no_grad():
            nb_sample += 1

            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            output_prob = torch.nn.Sigmoid()(predicted_masks)
            prob_np = output_prob.cpu().numpy()
            mask_np = ground_truth_masks.cpu().numpy()
            mask_final = np.where(mask_np == 1.0, 1.0, 0.0)

            gt.append(mask_final.flatten())
            pred.append(prob_np.flatten())

            prob_final = np.where(prob_np > threshold_value, 1.0, 0.0)

            correct += (torch.flatten(torch.tensor(prob_final)) == torch.flatten(torch.tensor(mask_final))).sum()
            total += len(torch.flatten(torch.tensor(prob_final)))

            jaccard += jaccard_score(np.squeeze(mask_final),
                                     np.squeeze(prob_final),
                                     average="micro")

    accuracy = correct.detach().numpy() / float(total)
    jaccard = jaccard / nb_sample
    dice_score = 2 * jaccard / (jaccard + 1.0)

    true_labels = np.concatenate(gt)
    pred_scores = np.concatenate(pred)

    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
    roc_auc = auc(fpr, tpr)
    eer, eer_threshold, eer_threshold_index = calculate_eer(fpr, tpr, thresholds)

    result_dict = {
        "accuracy": accuracy,
        "jaccard": jaccard,
        "dice": dice_score,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "eer_threshold_index": eer_threshold_index
    }

    save_path = os.path.join(os.path.dirname(model_path), "finetuned_sam_results.pkl")

    with open(save_path, 'wb') as f:
        pickle.dump(result_dict, f)

    print("File saved: ", save_path)

    print("Accuracy: ", accuracy)
    print("Jaccard Index: ", jaccard)
    print("Dice Score: ", dice_score)
    print("ROC-AUC: ", roc_auc)
    print("EER: ", eer)
    print("EER threshold: ", eer_threshold)

    return result_dict

def plot_roc_curve(result_dict):
    fpr = result_dict["fpr"]
    tpr = result_dict["tpr"]
    roc_auc = result_dict["roc_auc"]

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("Test.png")

if __name__ == "__main__":
    result_dict = test_model(data_directory=data_dir,
                             model_path=save_model_path,
                             batch_size=batch_size,
                             threshold_value=threshold_value)
    plot_roc_curve(result_dict)