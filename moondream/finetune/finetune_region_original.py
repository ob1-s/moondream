import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import math
from safetensors.torch import save_file
import datasets

from tqdm import tqdm
from bitsandbytes.optim import AdamW
import wandb

from ..torch.weights import load_weights_into_model
from ..torch.moondream import MoondreamModel, MoondreamConfig, text_encoder, EncodedImage, DEFAULT_MAX_OBJECTS
from ..torch.text import _produce_hidden
from ..torch.region import (
    decode_coordinate,
    decode_size,
    encode_coordinate,
    encode_size,
)

from PIL import Image, ImageDraw
import random

# This is a intended to be a basic starting point. Your optimal hyperparams and data may be different.
MODEL_PATH = "/kaggle/working/moondream/models/moondream_base.safetensors"
LR = 3e-5
EPOCHS = 8
GRAD_ACCUM_STEPS = 64


# def lr_schedule(step, max_steps):
#     x = step / max_steps
#     if x < 0.1:
#         return 0.1 * LR + 0.9 * LR * x / 0.1
#     else:
#         return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2
def lr_schedule(step, max_steps):
    x = step / max_steps
    warmup_frac = 0.2
    if x < warmup_frac:
        return 0.1 * LR + 0.9 * LR * x / warmup_frac
    else:
        decay_x = (x - warmup_frac) / (1 - warmup_frac)
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * decay_x)) / 2


def region_loss(
    hidden_states: torch.Tensor,
    w,
    labels: torch.Tensor,
    c_idx: torch.Tensor,
    s_idx: torch.Tensor,
):
    l_idx = torch.arange(len(labels))

    c_idx = c_idx - 1
    c_hidden = hidden_states[:, c_idx, :]
    c_logits = decode_coordinate(c_hidden, w)
    c_labels = labels[(l_idx % 4) < 2]

    c_loss = F.cross_entropy(
        c_logits.view(-1, c_logits.size(-1)),
        c_labels,
    )

    s_idx = s_idx - 1
    s_hidden = hidden_states[:, s_idx, :]
    s_logits = decode_size(s_hidden, w).view(-1, 1024)
    s_labels = labels[(l_idx % 4) >= 2]

    s_loss = F.cross_entropy(s_logits, s_labels)

    return c_loss + s_loss


def compute_iou(box1, box2):
    # box: [x_min, y_min, x_max, y_max]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_map(preds, gts, iou_threshold=0.5):
    """
    Simple mAP@IoU for single-class detections.
    preds: list, each either
      - a Tensor [N,4],
      - a list of Tensor([4]) or list of 4-floats,
      - OR a list of dicts like {'x_min':…, 'y_min':…, 'x_max':…, 'y_max':…}.
    gts:   list of Tensor [M,4].
    """
    def to_tensor(box_list):
        # box_list might be Tensor, or list of Tensor, or list of dicts, or list of lists
        if isinstance(box_list, torch.Tensor):
            return box_list.detach().cpu()
        if not isinstance(box_list, list):
            raise ValueError("Unexpected box_list type: %r" % type(box_list))
        if len(box_list)==0:
            return torch.zeros((0,4))
        # dict
        if isinstance(box_list[0], dict):
            coords = []
            for d in box_list:
                coords.append([
                    d['x_min'], d['y_min'],
                    d['x_max'], d['y_max']
                ])
            return torch.tensor(coords)
        # Tensor or list
        first = box_list[0]
        if isinstance(first, torch.Tensor):
            return torch.stack(box_list).detach().cpu()
        # assume list of floats
        return torch.tensor(box_list)

    all_precisions = []
    for pred_raw, true_tensor in zip(preds, gts):
        pred_tensor = to_tensor(pred_raw)
        true_boxes = true_tensor.detach().cpu().tolist()
        pred_boxes = pred_tensor.tolist()

        matched = set()
        tp = 0
        for pb in pred_boxes:
            # find best matching gt
            best_iou, best_j = 0, -1
            for j, tb in enumerate(true_boxes):
                if j in matched:
                    continue
                iou = compute_iou(pb, tb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold:
                tp += 1
                matched.add(best_j)

        fp = len(pred_boxes) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        all_precisions.append(precision)

    return sum(all_precisions) / len(all_precisions) if all_precisions else 0.0


def eval_detect(dataset, eval_idxs, model):
    model.eval()
    preds, gts = [], []
    first_idx = eval_idxs[0]

    for idx in eval_idxs:
        sample = dataset[idx]

        sample_class = sample["class_names"][0].replace('-', ' ')
        print(f"RUNNING EVAL for class `{sample_class}`")

        # 1) call the native helper instead of manual cache‐merge
        result = model.detect(
            sample["image"],
            sample_class,
            settings={"max_objects": DEFAULT_MAX_OBJECTS},
        )
        objs = result["objects"]  # list of {x_min, y_min, x_max, y_max}

        print("RESULT", str(objs))

        preds.append(objs)
        gts.append(sample["boxes"])

        # 2) on the very first eval sample, draw & log its predictions
        if idx == first_idx:
            vis = sample["image"].convert("RGB").copy()
            draw = ImageDraw.Draw(vis)
            for o in objs:
                draw.rectangle(
                    [o["x_min"], o["y_min"], o["x_max"], o["y_max"]],
                    outline="red",
                    width=2,
                )
            wandb.log({
                "eval/example_detect":
                    wandb.Image(vis, caption="Detect via native API")
            })

    return compute_map(preds, gts)


class GroundedDetection(Dataset):
    def __init__(self, split: str = "train"):
        self.dataset: datasets.Dataset = datasets.load_dataset(
            "oliveirabruno01/grounded-detection", split=split
        )

        self.dataset = self.dataset.filter(lambda row: len(row["bboxes"]) > 0)
        self.dataset = self.dataset.shuffle(seed=3301)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["scene_image"]
        bboxes = row["bboxes"]
        labels = [row["asset_name"] for _ in row["bboxes"]]

        # convert to YOLO format
        norm_boxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            w_img, h_img = image.size
            x_c = (x_min + x_max) / 2 / w_img
            y_c = (y_min + y_max) / 2 / h_img
            w_n = (x_max - x_min) / w_img
            h_n = (y_max - y_min) / h_img
            norm_boxes.append([x_c, y_c, w_n, h_n])

        # group by label
        objects = {}
        for box, label in zip(norm_boxes, labels):
            objects.setdefault(label, []).append(box)

        flat_boxes = []
        class_names = []
        for label, box_list in objects.items():
            for b in box_list:
                flat_boxes.append(b)
                class_names.append(label)

        flat_boxes = torch.as_tensor(flat_boxes, dtype=torch.float16)
        image_id = torch.tensor([idx], dtype=torch.int64)

        return {
            "image": image,
            "boxes": flat_boxes,
            "class_names": class_names,
            "image_id": image_id,
        }


def main():
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    wandb.init(
        project="moondream-ft",
        config={
            "EPOCHS": EPOCHS,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
        },
    )

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(MODEL_PATH, model)

    for p in model.vision.parameters(): p.requires_grad = False
    for p in model.text.parameters():   p.requires_grad = False

    # If you are struggling with GPU memory, try AdamW8Bit
    optimizer = AdamW(
        [{"params": model.region.parameters()}],
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    dataset = GroundedDetection()
    print(f"GroundedDetection: {len(dataset)} items.")
    idxs = list(range(len(dataset)))
    train_idxs = idxs[:2048]
    eval_idxs = idxs[2048:2048+128]

    total_steps = EPOCHS * len(train_idxs) // GRAD_ACCUM_STEPS
    pbar = tqdm(total=total_steps)

    # pre‐finetune eval
    mAP0 = eval_detect(dataset, eval_idxs, model)
    wandb.log({"mAP_v1": mAP0})
    model.train()

    random.seed(3301)

    i = 0
    for epoch in range(EPOCHS):
        if epoch:
            random.shuffle(train_idxs)
          
        for sample_idx in train_idxs:
            sample = dataset[sample_idx]
            i += 1

            torch.cuda.empty_cache()

            with torch.no_grad():
                img_emb = model._run_vision_encoder(sample["image"])
                bos_emb = text_encoder(
                    torch.tensor(
                        [[model.config.tokenizer.bos_id]], device=model.device
                    ),
                    model.text,
                )
                eos_emb = text_encoder(
                    torch.tensor(
                        [[model.config.tokenizer.eos_id]], device=model.device
                    ),
                    model.text,
                )

            boxes_by_class = {}
            for box, cls in zip(sample["boxes"], sample["class_names"]):
                boxes_by_class.setdefault(cls, []).append(box)

            total_loss = 0.0
            for class_name, boxes_list in boxes_by_class.items():
                with torch.no_grad():
                    instruction = f"\n\nDetect: {class_name.replace('-', ' ')}\n\n"
                    instruction_tokens = model.tokenizer.encode(instruction).ids
                    instruction_emb = text_encoder(
                        torch.tensor([[instruction_tokens]], device=model.device),
                        model.text,
                    ).squeeze(0)

                cs_emb = []
                cs_labels = []
                c_idx = []
                s_idx = []
                for bb in boxes_list:
                    l_cs = len(cs_emb)
                    cs_emb.extend(
                        [
                            encode_coordinate(bb[0].unsqueeze(0), model.region),
                            encode_coordinate(bb[1].unsqueeze(0), model.region),
                            encode_size(bb[2:4], model.region),
                        ]
                    )
                    c_idx.extend([l_cs, l_cs + 1])
                    s_idx.append(l_cs + 2)

                    # Create coordinate bin labels - unchanged
                    coord_labels = [
                        int(min(max(torch.round(p * 1023), 0), 1023).item()) for p in bb[:2]
                    ]

                    # Create size bin labels using log-scale mapping
                    s_log2_bins = []
                    for s_val in bb[2:4]:
                        s_val = float(s_val)
                        s_clamped = max(s_val, 1 / 1024)
                        s_log2 = math.log2(s_clamped)
                        mapped = (s_log2 + 10.0) / 10.0 * 1023.0
                        s_bin = int(round(mapped))
                        s_bin = max(min(s_bin, 1023), 0)
                        s_log2_bins.append(s_bin)

                    # Combine coordinate and size bin labels
                    cs_labels.extend(coord_labels + s_log2_bins)

                if len(cs_emb) == 0:
                    continue
                cs_emb = torch.stack(cs_emb)

                inputs_embeds = torch.cat(
                    [bos_emb, img_emb[None], instruction_emb, cs_emb[None], eos_emb],
                    dim=1,
                )
                prefix = inputs_embeds.size(1) - cs_emb.size(0)
                c_idx = torch.tensor(c_idx) + prefix
                s_idx = torch.tensor(s_idx) + prefix

                hidden = _produce_hidden(
                    inputs_embeds=inputs_embeds, w=model.text, config=config.text
                )

                loss = region_loss(
                    hidden_states=hidden,
                    w=model.region,
                    labels=torch.tensor(cs_labels, dtype=torch.int64),
                    c_idx=c_idx,
                    s_idx=s_idx,
                )
                total_loss += loss

            total_loss.backward()

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr_val = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_val
                pbar.set_postfix(
                    {"step": i // GRAD_ACCUM_STEPS, "loss": total_loss.item()}
                )
                pbar.update(1)
                wandb.log(
                    {
                        "loss/train": total_loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
        
        # post‐epoch eval
        mAP_e = eval_detect(dataset, eval_idxs, model)
        wandb.log({"mAP_v1": mAP_e})
        model.train()
      
    wandb.finish()

    # Replace with your desired output location.
    save_file(
        model.state_dict(),
        "moondream_finetune_original.safetensors",
    )


if __name__ == "__main__":
    """
    Replace paths with your appropriate paths.
    To run: python -m moondream.finetune.finetune_region

    1 epoch of fine-tuning on the example 'Waste Detection' dataset results in an
    increase in mAP from 61.82 to 69.82.
    """
    main()
