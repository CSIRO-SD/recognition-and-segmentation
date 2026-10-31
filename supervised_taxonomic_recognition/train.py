import os
import json
import logging
import timm
from collections import defaultdict
from tqdm import tqdm
from torchvision.models import vit_b_16, ViT_B_16_Weights

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import Swin_T_Weights, swin_t
from pycocotools.coco import COCO
from PIL import Image
from timm.scheduler import CosineLRScheduler

# ------------------------------------------------------------------------------
# 1. 单标签 COCO Dataset：每张图返回 (img_tensor, label_idx)
# ------------------------------------------------------------------------------
class CocoSingleLabelDataset(Dataset):
    def __init__(self, img_root, ann_file, transform=None):
        """
        img_root:   图片根目录
        ann_file:   COCO-format annotation JSON
        transform:  torchvision transforms
        """
        self.img_root = img_root
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        # COCO JSON 里所有 category_id，sorted 保证顺序一致
        self.cat_ids = sorted(self.coco.getCatIds())
        self.num_classes = len(self.cat_ids)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img_path = info['file_name']
        img = Image.open(os.path.join(self.img_root, img_path)).convert('RGB')
        if self.transform:
            img = self.transform(img)

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        if len(anns) == 0:
            label_idx = 0
        else:
            # 取第一个 annotation 的类别作为标签
            cid = anns[0]['category_id']
            label_idx = self.cat_ids.index(cid)

        return img, torch.tensor(label_idx, dtype=torch.long)

# ------------------------------------------------------------------------------
# 2. 超参 & Transform
# ------------------------------------------------------------------------------
# TRAIN_IMG_ROOT      = '/datasets/work/icvg-phd-marsfield/work/Houke/final_data/train2017'
# TRAIN_ANN_FILE      = '/datasets/work/icvg-phd-marsfield/work/Houke/final_data/annotations/instances_train2017.json'
# VAL_IMG_ROOT      = '/datasets/work/icvg-phd-marsfield/work/Houke/final_data/val2017'
# VAL_ANN_FILE      = '/datasets/work/icvg-phd-marsfield/work/Houke/final_data/annotations/instances_val2017.json'
TRAIN_IMG_ROOT      = '/datasets/work/d61-insect-digitisation/work/Dataset/B-DigiVol_images/final_data/our_dataset_subset_train_val/LabInsect-48K-train-images-subset'
TRAIN_ANN_FILE      = '/datasets/work/d61-insect-digitisation/work/Dataset/B-DigiVol_images/final_data/our_dataset_subset_train_val/LabInsect-48K-train-subset.json'
VAL_IMG_ROOT      = '/datasets/work/d61-insect-digitisation/work/Dataset/B-DigiVol_images/final_data/our_dataset_subset_train_val/LabInsect-48K-test-images-subset'
VAL_ANN_FILE      = '/datasets/work/d61-insect-digitisation/work/Dataset/B-DigiVol_images/final_data/our_dataset_subset_train_val/LabInsect-48K-test-subset.json'
BATCH_SIZE    = 32
BASE_LR            = 5e-3
EPOCHS        = 50
DEVICE        = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


image_ids_annotations = defaultdict(list)
file = open(TRAIN_ANN_FILE)
anns = json.load(file)
image_ids = list()

# Add into datastructure
for ann in anns['annotations']:
    image_id = ann['image_id'] # Are integers
    image_ids.append(image_id)
    image_ids_annotations[image_id].append(ann)
# 初始化每个类别的计数（包含 1~643）
catergory_id_count = {i: 0 for i in range(1, 644)}

# 统计每个 annotation 出现的次数
for ann in anns['annotations']:
    cid = ann['category_id']
    # 如果有意外的 cid（超出 1~643），可以先过滤一下
    if cid==0:
        print(cid)
    if 1 <= cid <= 643:
        catergory_id_count[cid] += 1

print(catergory_id_count)
# Get mapping category_id to category name
catergory_id_to_name = dict()

for index, cat in enumerate(anns['categories']):
    catergory_id_to_name[cat['id']] = cat['name']
# 1. 取出所有类别 ID，并排序（保证固定顺序）
category_ids = sorted([cat['id'] for cat in anns['categories']])

# 2. 按照同一顺序构造 values 与 classes
values = [catergory_id_count.get(cid, 0) for cid in category_ids]
classes = [catergory_id_to_name[cid] for cid in category_ids]

transform_train = transforms.Compose([
    transforms.Resize((512, 512)),
    # transforms.RandomResizedCrop(512),            # 输入大小 512
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # 控制对比度、亮度
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

transform_val = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
])

# ------------------------------------------------------------------------------
# 3. 构建 Dataset & DataLoader
# ------------------------------------------------------------------------------
train_dataset      = CocoSingleLabelDataset(TRAIN_IMG_ROOT, TRAIN_ANN_FILE, transform_train)

val_dataset      = CocoSingleLabelDataset(VAL_IMG_ROOT, VAL_ANN_FILE, transform_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=10)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=10)

# ------------------------------------------------------------------------------
# 4. 定义模型：Swin-Tiny + 新的 head
# ------------------------------------------------------------------------------

# MODEL_NAME = 'vit_base_patch16_224'
MODEL_NAME = 'resnet50'
# MODEL_NAME = 'hrnet_w32'
# MODEL_NAME = 'mobilenetv4_large'

# 把 patch_embed 的 img_size 换成 512
if MODEL_NAME.startswith('vit'):
    model = timm.create_model(
        MODEL_NAME,  # 这里 vit_base_patch16_224 表示 ViT-B/16
        pretrained=True,
        num_classes=train_dataset.num_classes,  # 643
        img_size=512                     # 让 patch-embed 接受 512×512 输入
    )
else:
    model = timm.create_model(
        MODEL_NAME, 
        pretrained=True,
        num_classes=train_dataset.num_classes  # 你的类别数
    )

model = model.to(DEVICE)

# ------------------------------------------------------------------------------
# 5. 损失 & 优化器
# ------------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
LR = BASE_LR * BATCH_SIZE /512
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
#optimizer.load_state_dict(ckpt['optimizer_state'])


warmup_epochs = 5

scheduler = CosineLRScheduler(
    optimizer,
    t_initial=EPOCHS - warmup_epochs,
    lr_min=1e-6,
    warmup_lr_init=1e-6,
    warmup_t=warmup_epochs,
    cycle_limit=1,
    t_in_epochs=True,
)

# ------------------------------------------------------------------------------
# 6. 训练 & 验证 函数
# ------------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, scheduler):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc='Train', leave=False)
    for step, (imgs, labels) in enumerate(loop):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)                 # [B, 643]
        loss    = criterion(outputs, labels)  # CrossEntropyLoss expects [B], long
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(avg_loss=running_loss/(loop.n + 1))

        scheduler.step_update(epoch * len(train_loader) + step)
    return running_loss / len(loader)

def validate(model, loader, criterion, classes):
    model.eval()
    running_loss = 0.0

    # Top-1 准确率统计
    correct_samples = 0
    total_samples   = 0

    # 每类 TP/FP/FN/TN 统计
    tp = {c:0 for c in classes}
    fp = {c:0 for c in classes}
    fn = {c:0 for c in classes}
    tn = {c:0 for c in classes}

    with torch.no_grad():
        loop = tqdm(loader, desc='Val  ', leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)                   # [B, C]
            loss    = criterion(outputs, labels)
            running_loss += loss.item()

            # Top-1 预测
            preds = outputs.argmax(dim=1)           # [B]
            correct_samples += (preds == labels).sum().item()
            total_samples   += labels.size(0)

            # 拉回 CPU 方便后续循环
            preds_cpu  = preds.cpu().tolist()
            labels_cpu = labels.cpu().tolist()

            # 堆叠每个样本、每个类别的 TP/FP/FN/TN
            for p, t in zip(preds_cpu, labels_cpu):
                for idx, cname in enumerate(classes):
                    if   t == idx and p == idx: tp[cname] += 1
                    elif t != idx and p == idx: fp[cname] += 1
                    elif t == idx and p != idx: fn[cname] += 1
                    else:                         tn[cname] += 1

    avg_loss = running_loss / len(loader)
    top1_acc = correct_samples / total_samples if total_samples>0 else 0

    # 计算每个类别的 Precision/Recall/F1/Accuracy
    per_class = {}
    for cname in classes:
        _tp, _fp, _fn, _tn = tp[cname], fp[cname], fn[cname], tn[cname]
        denom = _tp + _fp + _fn + _tn
        if denom == 0:
            continue
        acc  = (_tp + _tn) / denom
        prec = _tp / (_tp + _fp) if (_tp + _fp)>0 else 0
        rec  = _tp / (_tp + _fn) if (_tp + _fn)>0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        per_class[cname] = {'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1}

    # Macro-average
    n_cls = len(classes)
    macro_acc  = sum(v['acc']  for v in per_class.values()) / n_cls
    macro_prec = sum(v['prec'] for v in per_class.values()) / n_cls
    macro_rec  = sum(v['rec']  for v in per_class.values()) / n_cls
    macro_f1   = sum(v['f1']   for v in per_class.values()) / n_cls

    # Micro-average
    sum_tp = sum(tp.values())
    sum_fp = sum(fp.values())
    sum_fn = sum(fn.values())
    sum_tn = sum(tn.values())
    micro_acc  = (sum_tp + sum_tn) / (sum_tp+sum_fp+sum_fn+sum_tn) \
                 if (sum_tp+sum_fp+sum_fn+sum_tn)>0 else 0
    micro_prec = sum_tp / (sum_tp + sum_fp) if (sum_tp+sum_fp)>0 else 0
    micro_rec  = sum_tp / (sum_tp + sum_fn) if (sum_tp+sum_fn)>0 else 0
    micro_f1   = 2*micro_prec*micro_rec/(micro_prec+micro_rec) \
                 if (micro_prec+micro_rec)>0 else 0

    # 打印所有指标
    print(f"Val Loss: {avg_loss:.4f} | Top-1 Acc: {top1_acc:.4f}")
    print(f"Macro Acc: {macro_acc:.4f} | Prec: {macro_prec:.4f} | Rec: {macro_rec:.4f} | F1: {macro_f1:.4f}")
    print(f"Micro Acc: {micro_acc:.4f} | Prec: {micro_prec:.4f} | Rec: {micro_rec:.4f} | F1: {micro_f1:.4f}")

    return avg_loss, per_class, top1_acc, (macro_acc, macro_prec, macro_rec, macro_f1), \
           (micro_acc, micro_prec, micro_rec, micro_f1)

# ------------------------------------------------------------------------------
# 7. 主训练流程
# ------------------------------------------------------------------------------
save_dir = os.path.join("checkpoints", MODEL_NAME)
os.makedirs(save_dir, exist_ok=True)
log_path = os.path.join(save_dir, 'log.txt')

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler)
    scheduler.step(epoch + 1)

    val_loss, per_class_results, top1_acc, macro_metrics, micro_metrics = validate(model, val_loader, criterion, classes)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    msg = (
        f"Epoch {epoch+1}/{EPOCHS}\n"
        f"Top-1 Acc: {top1_acc:.4f}\n"
        f"Macro Acc: {macro_metrics[0]:.4f} | Prec: {macro_metrics[1]:.4f} "
        f"| Rec: {macro_metrics[2]:.4f} | F1: {macro_metrics[3]:.4f}\n"
        f"Micro Acc: {micro_metrics[0]:.4f} | Prec: {micro_metrics[1]:.4f} "
        f"| Rec: {micro_metrics[2]:.4f} | F1: {micro_metrics[3]:.4f}\n"
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
        + "-"*60 + "\n"
    )

    with open(log_path, "a") as f:
        f.write(msg)

    # 保存检查点
    ckpt_path = os.path.join(save_dir, "checkpoint_latest.pth")
    torch.save(model.state_dict(), ckpt_path)
