#@title training code
import torch
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
import math
from safetensors.torch import save_file
import datasets

from tqdm import tqdm
from bitsandbytes.optim import AdamW, AdamW8bit
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

from datasets import Sequence
from datasets import Image as DsImage

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
    for pred_raw, true_raw in zip(preds, gts):
        pred_tensor = to_tensor(pred_raw)
        true_tensor = to_tensor(true_raw)

        pred_boxes = pred_tensor.tolist()
        true_boxes = true_tensor.tolist()

        matched = set()
        tp = 0
        for pb in pred_boxes:
            best_iou, best_j = 0, -1
            for j, tb in enumerate(true_boxes):
                if j in matched: continue
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


def eval_detect_ref(dataset, eval_idxs, model):
    model.eval()
    preds, gts = [], []
    first_idx = eval_idxs[0]

    for idx in eval_idxs:
        sample = dataset[idx]

        # 1) call the native helper instead of manual cache‐merge
        result = model.detect_with_reference(
            sample["image"],
            sample["reference"],
            settings={"max_objects": DEFAULT_MAX_OBJECTS},
        )
        objs = result["objects"]  # list of {x_min, y_min, x_max, y_max}

        gt_boxes = []
        # sample["boxes"] is now normalized [x_min, y_min, width, height]
        for x_min_n, y_min_n, w_n, h_n in sample["boxes"].detach().cpu().tolist():
             # Convert to normalized [x_min, y_min, x_max, y_max] for mAP calculation
             # (Assuming model.detect outputs this format and compute_iou expects it)
             x_max_n = x_min_n + w_n
             y_max_n = y_min_n + h_n
             # Clamp to be safe, although clamping in __getitem__ might be sufficient
             x_min_n = max(0.0, min(1.0, x_min_n))
             y_min_n = max(0.0, min(1.0, y_min_n))
             x_max_n = max(0.0, min(1.0, x_max_n))
             y_max_n = max(0.0, min(1.0, y_max_n))
             gt_boxes.append([x_min_n, y_min_n, x_max_n, y_max_n])
        
        preds.append(objs)      # objs is already a list of dicts with normalized corners
        gts.append(gt_boxes)    # now a list of 4‑floats matching the preds format

        # 2) on the very first eval sample, draw & log its predictions
        for o in objs:
                # normalized coords:
                x_min_n, y_min_n = o["x_min"], o["y_min"]
                x_max_n, y_max_n = o["x_max"], o["y_max"]
            
                # convert to pixels
                x0 = x_min_n * w_img
                y0 = y_min_n * h_img
                x1 = x_max_n * w_img
                y1 = y_max_n * h_img
            
                draw.rectangle(
                    [x0, y0, x1, y1],
                    outline="red",
                    width=2,
                )
            
            # Draw Ground Truth boxes (Green) - CORRECTED
            for bb in sample["boxes"]:
                # bb is now [x_min_n, y_min_n, width_n, height_n]
                x_min_n, y_min_n, width_n, height_n = bb.detach().cpu().tolist()

                # Convert normalized [xmin, ymin, width, height] to pixel [xmin, ymin, xmax, ymax]
                x_min_px = x_min_n * w_img
                y_min_px = y_min_n * h_img
                x_max_px = (x_min_n + width_n) * w_img
                y_max_px = (y_min_n + height_n) * h_img

                # Optional: Clamp pixel coordinates to image boundaries for robustness
                x_min_px = max(0, min(w_img - 1, x_min_px))
                y_min_px = max(0, min(h_img - 1, y_min_px))
                x_max_px = max(0, min(w_img - 1, x_max_px))
                y_max_px = max(0, min(h_img - 1, y_max_px))

                draw.rectangle(
                    [x_min_px, y_min_px, x_max_px, y_max_px],
                    outline="green",
                    width=2,
                )
            wandb.log({
                "eval/example_detectref":
                    wandb.Image(vis, caption="Detect<REF> via native API")
            })

    return compute_map(preds, gts)


def collate_references(ref_images):
    """
    Collate a list of PIL Images (possibly RGBA) into a single prototype image
    with a solid white background, arranged in a row.
    """
    if not ref_images:
        raise ValueError("No reference images provided.")

    # Make sure all images are RGBA so we can composite
    rgba_refs = [img.convert("RGBA") for img in ref_images]

    widths, heights = zip(*(img.size for img in rgba_refs))
    total_width = sum(widths)
    max_height = max(heights)

    # Create a solid-white RGBA canvas
    prototype = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 255))

    # Paste each reference using its alpha channel
    x_offset = 0
    for img in rgba_refs:
        prototype.paste(img, (x_offset, 0), mask=img)
        x_offset += img.width

    # If you want an RGB image back (no alpha), flatten to white
    return prototype.convert("RGB")


class GroundedDetection(Dataset):
    def __init__(self, split: str = "train"):
        self.dataset: datasets.Dataset = datasets.load_dataset(
            "oliveirabruno01/grounded-detection", split=split
        )

        self.dataset = self.dataset.filter(lambda row: len(row["bboxes"]) > 0)
        self.dataset = self.dataset.cast_column("reference_images", Sequence(DsImage()))

        def make_proto(example):
            example["reference_image"] = collate_references(example["reference_images"])
            return example

        self.dataset = self.dataset.map(
            make_proto,
            batched=False,
            remove_columns=["reference_images"],   # drop original list
        )

        self.dataset = self.dataset.shuffle(seed=3301)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["scene_image"]
        ref_img = row["reference_image"]
        bboxes = row["bboxes"]
        labels = [row["asset_name"] for _ in row["bboxes"]]

        # Calculate normalized top-left corner
            x_min_norm = x_min_px / w_img
            y_min_norm = y_min_px / h_img

            # Calculate normalized width and height
            width_norm = (x_max_px - x_min_px) / w_img
            height_norm = (y_max_px - y_min_px) / h_img

            # Optional: Clamp values to [0.0, 1.0] to avoid numerical issues
            x_min_norm = max(0.0, min(1.0, x_min_norm))
            y_min_norm = max(0.0, min(1.0, y_min_norm))
            width_norm = max(0.0, min(1.0, width_norm))
            height_norm = max(0.0, min(1.0, height_norm))

            # Append in the correct [x_min, y_min, width, height] format
            norm_boxes.append([x_min_norm, y_min_norm, width_norm, height_norm])

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
            "reference": ref_img,
            "boxes": flat_boxes,
            "class_names": class_names,
            "image_id": image_id,
        }


def main():
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    wandb.init(
        project="moondream-ft",
        name="detect-ref-concat",
        config={
            "EPOCHS": EPOCHS,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
        },
    )

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(MODEL_PATH, model, device="cuda:0")

    for p in model.vision.parameters(): p.requires_grad = False
    for p in model.text.parameters():   p.requires_grad = False
    for p in model.vision.proj_mlp.fc1.parameters(): p.requires_grad = True
    for p in model.vision.proj_mlp.fc2.parameters(): p.requires_grad = True

    # for name, module in model.vision.named_modules():
    #   print(name, type(module))
    #   print("----- parameters -----")
    # for name, param in model.vision.named_parameters():
    #   print(name, param.shape)

    # If you are struggling with GPU memory, try AdamW8Bit
    optimizer = AdamW8bit(
        [
            {"params": model.region.parameters(), "lr": LR},
            {"params": model.vision.proj_mlp.fc1.parameters(), "lr": LR},
            {"params": model.vision.proj_mlp.fc2.parameters(), "lr": LR},
        ],
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    # _ds = WasteDetection()
    # --- split dataset ---
    dataset = GroundedDetection()
    print(f"GroundedDetection: {len(dataset)} items.")
    idxs = list(range(len(dataset)))
    train_idxs = idxs[:2048]
    eval_idxs = idxs[2048:2048+128]

    total_steps = EPOCHS * len(train_idxs) // GRAD_ACCUM_STEPS
    pbar = tqdm(total=total_steps)

    # pre‐finetune eval
    mAP0 = eval_detect_ref(dataset, eval_idxs, model)
    wandb.log({"mAP_v1": mAP0})
    model.train()

    random.seed(3301)

    i = 0
    for epoch in range(EPOCHS):
        if epoch:
            random.shuffle(train_idxs)

        frac_class_epoch = 0.8 * (1 - epoch / (EPOCHS - 1))
        wandb.log({"frac_class_epoch": frac_class_epoch}, step=epoch)
        
        for sample_idx in train_idxs:
            sample = dataset[sample_idx]
            i += 1

            torch.cuda.empty_cache()

            # 1) Compute vision embeddings for scene + reference
            img_emb = model._run_vision_encoder(sample["image"])      # [D]
            ref_emb = model._run_vision_encoder(sample["reference"])  # [D]

            with torch.no_grad():
                # 2) Build the shared text prefix: [BOS][IMG][REF]
                bos_emb = text_encoder(
                    torch.tensor([[model.config.tokenizer.bos_id]], device=model.device),
                    model.text,
                )                        # [1,1,D]
                eos_emb = text_encoder(
                    torch.tensor([[model.config.tokenizer.eos_id]], device=model.device),
                    model.text,
                )                        # [1,1,D]

            # note: instruction and region tokens come after these
            # so prefix_len = 1 (BOS) + 1 (IMG) + 1 (REF) + instr_len
            prefix_base = 1 + 1 + 1

            # Group ground‐truth boxes by “class” (we still use class_names, though
            # for DetectRef that token is just a placeholder)
            boxes_by_class = {}
            for box, cls in zip(sample["boxes"], sample["class_names"]):
                boxes_by_class.setdefault(cls, []).append(box)

            total_loss = 0.0

            # 3) For each “class” (in DetectRef you can ignore the string, we just loop)
            for _cls, boxes_list in boxes_by_class.items():
                use_class = (random.random() < frac_class_epoch)
                
                with torch.no_grad():
                    instruction = f"\n\nDetect {'<REF>' if not use_class else _cls.replace('-', ' ')}\n\n"
                    instruction_tokens = model.tokenizer.encode(instruction).ids
                    instruction_emb = text_encoder(
                        torch.tensor([[instruction_tokens]], device=model.device),
                        model.text,
                    ).squeeze(0)              # [instr_len, D]

                # Now prefix_len = base + instr_len
                prefix_len = prefix_base + instruction_emb.size(0)

                # 3b) Build your region token embeddings (coord + size)
                cs_emb   = []
                cs_labels= []
                c_idx    = []
                s_idx    = []

                for bb in boxes_list:
                    L = len(cs_emb)
                    # encode x_min, x_max, width/height
                    cs_emb.extend([
                        encode_coordinate(bb[0].unsqueeze(0), model.region),
                        encode_coordinate(bb[1].unsqueeze(0), model.region),
                        encode_size(bb[2:4],       model.region),
                    ])
                    c_idx += [L, L+1]
                    s_idx.append(L+2)

                    # make the GT bins in the same order
                    # coords
                    coord_bins = [ int((p*1023).clamp(0,1023).item()) for p in bb[:2] ]
                    # sizes via log mapping
                    size_bins = []
                    for val in bb[2:4]:
                        lv = float(val)
                        mapped = (math.log2(max(lv,1/1024))+10.0)/10.0*1023.0
                        size_bins.append(int(min(max(round(mapped),0),1023)))
                    cs_labels += coord_bins + size_bins

                if not cs_emb:
                    continue

                cs_emb = torch.stack(cs_emb)    # [num_tokens, D]

                # 4) Concatenate everything: [BOS] [IMG] [REF] [INST...] [CS...] [EOS]
                inputs_embeds = torch.cat([
                    bos_emb,           # [1,1,D]
                    img_emb[None],     # [1,1,D]
                    ref_emb[None],     # [1,1,D]
                    instruction_emb,   # already [1, instr_len, D]
                    cs_emb[None],      # [1, num_cs_tokens, D]
                    eos_emb,           # [1,1,D]
                ], dim=1)              # → [1, seq_len, D]

                # 5) Compute hidden states via your text stack
                hidden = _produce_hidden(
                    inputs_embeds=inputs_embeds,
                    w=model.text,
                    config=config.text,
                )  # [1, seq_len, D]

                # 6) Shift your indices to point into `hidden`:
                #    region tokens start at `prefix_len`
                c_idx = torch.tensor(c_idx, device=model.device) + prefix_len
                s_idx = torch.tensor(s_idx, device=model.device) + prefix_len

                # 7) Compute and accumulate region loss
                loss = region_loss(
                    hidden_states=hidden,
                    w=model.region,
                    labels=torch.tensor(cs_labels, device=model.device, dtype=torch.int64),
                    c_idx=c_idx,
                    s_idx=s_idx,
                )
                total_loss += loss

            # 8) Backprop + optimizer step
            total_loss.backward()
            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                # update lr
                lr_val = lr_schedule(i // GRAD_ACCUM_STEPS, total_steps)
                optimizer.param_groups[0]["lr"] = lr_val

                pbar.update(1)
                pbar.set_postfix(step=i//GRAD_ACCUM_STEPS, loss=total_loss.item())
                wandb.log({"loss/train": total_loss.item(), "lr": lr_val})

        # post‐epoch eval
        mAP_e = eval_detect_ref(dataset, eval_idxs, model)
        wandb.log({"mAP_v1": mAP_e})
        model.train()

    wandb.finish()

    # Replace with your desired output location.
    save_file(
        model.state_dict(),
        "moondream_finetune_v1.safetensors",
    )


if __name__ == "__main__":
    """
    Replace paths with your appropriate paths.
    To run: python -m moondream.finetune.finetune_region

    1 epoch of fine-tuning on the example 'Waste Detection' dataset results in an
    increase in mAP from 61.82 to 69.82.
    """
    main()
