import json
import torch
from torchmetrics import AveragePrecision as AP

def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)

gt_path = '/home/scanar/sages_challenge/SurgLatentGraph/results/sages_og_preds/test/lg/ds_gts.json'
pred_path = '/home/scanar/sages_challenge/SurgLatentGraph/results/sages_og_preds/test/lg/ds_preds.json'

gt_dict = load_json(gt_path)
pred_dict = load_json(pred_path)

# Ordenar por nombre del frame
gt_items = sorted(gt_dict.items(), key=lambda x: x[0])   # x[0] es el nombre del frame
pred_items = sorted(pred_dict.items(), key=lambda x: x[0])

# Separar etiquetas
gt = torch.tensor([x[1] for x in gt_items])      # x[1] es la lista de etiquetas
preds = torch.tensor([x[1] for x in pred_items])
# Evaluar AP
torch_ap = AP(task='multilabel', num_labels=3, average='none')
ap_result = torch_ap(preds, gt)

print("Average Precision por clase:", ap_result)

