#@title training code (inline REF emb)
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


def eval_detect_inline(dataset, eval_idxs, model):
    model.eval()
    preds, gts = [], []
    first_idx = eval_idxs[0]
    
    # grab the precomputed prefix / suffix id lists from your tokenizer
    with torch.no_grad():
        prefix_ids = model.config.tokenizer.templates["detect"]["prefix"]+[220]
        suffix_ids = model.config.tokenizer.templates["detect"]["suffix"]

        prefix_emb = text_encoder(
            torch.tensor([prefix_ids], device=device),
            model.text,
        )  # Shape: [1, prefix_len, D]

        suffix_emb = text_encoder(
            torch.tensor([suffix_ids], device=device),
            model.text,
        )  # Shape: [1, suffix_len, D]

        bos_emb = text_encoder(
            torch.tensor([[model.config.tokenizer.bos_id]], device=device),
            model.text,
        ) # Shape: [1, 1, D]

    for idx in eval_idxs:
        sample = dataset[idx]

        with torch.no_grad():
            img_emb_flat = model._run_vision_encoder(sample["image"])      # [D]
            ref_emb_flat = model._run_vision_encoder(sample["reference"])  # [D]
            img_emb = img_emb_flat.to(device).unsqueeze(0).unsqueeze(0)    # [1, 1, D]
            ref_emb = ref_emb_flat.to(device).unsqueeze(0).unsqueeze(0)    # [1, 1, D]

            # 2. Construct Full Prompt Embedding (Mirrors Training Order)
            # [BOS] [SCENE_IMG] [PREFIX] [REF_IMG] [SUFFIX]
            full_prompt_emb = torch.cat([
                bos_emb, prefix_emb, ref_emb, suffix_emb
            ], dim=1)
            total_prompt_len = full_prompt_emb.size(1)

            # 3. Manual Forward Pass through Transformer (Replaces _prefill_prompt)
            # --- Reset KV Cache ---
            if hasattr(model.text, 'blocks') and model.text.blocks:
                for block in model.text.blocks:
                     if hasattr(block, 'kv_cache') and block.kv_cache is not None:
                         if hasattr(block.kv_cache, 'k_cache'): block.kv_cache.k_cache.zero_()
                         if hasattr(block.kv_cache, 'v_cache'): block.kv_cache.v_cache.zero_()

            # --- Manual Transformer Forward Pass ---
            hidden_states = full_prompt_emb
            position_ids = torch.arange(0, total_prompt_len, device=device).unsqueeze(0)
            for block in model.text.blocks:
                 block_output = block(
                     hidden_states,
                     use_cache=True,
                     position_ids=position_ids[:, :hidden_states.size(1)],
                 )
                 hidden_states = block_output[0]

            hidden_states = model.text.norm(hidden_states)
            last_hidden = hidden_states[:, -1:, :] # State after full prompt

            # --- Predict Initial Token for Generation ---
            next_logits = model.text.lm_head(last_hidden)
            initial_next_token_id = torch.argmax(next_logits, dim=-1)

            # 4. Generate Region Points (Replaces original call)
            gen_pos = total_prompt_len # Start generation after the prompt
            objs = model._generate_points(
                last_hidden,
                next_token=initial_next_token_id, # Use predicted token
                pos=gen_pos,
                include_size=True,
                max_objects=DEFAULT_MAX_OBJECTS
            )

        # --- Ground Truth Processing (Same as your corrected version) ---
        gt_boxes = []
        for box_data in sample["boxes"].detach().cpu().tolist():
             x_min_n, y_min_n, w_n, h_n = box_data
             x_max_n = x_min_n + w_n
             y_max_n = y_min_n + h_n
             x_min_n = max(0.0, min(1.0, x_min_n))
             y_min_n = max(0.0, min(1.0, y_min_n))
             x_max_n = max(0.0, min(1.0, x_max_n))
             y_max_n = max(0.0, min(1.0, y_max_n))
             gt_boxes.append([x_min_n, y_min_n, x_max_n, y_max_n]) # Format for compute_map

        preds.append(objs)
        gts.append(gt_boxes)

        # --- Visualization (Same as your corrected version) ---
        if idx == first_idx:
            # Use .get for safer access to class_names
            sample_class = sample.get("class_names", ["unknown"])[0].replace('-', ' ')
            print(f"\nRUNNING EVAL (Replacement) for class placeholder: `{sample_class}`")
            print("RESULT", str(objs))
            print(f"EXPECTED (norm xywh): {sample['boxes']}")

            vis = sample["image"].convert("RGB").copy()
            draw = ImageDraw.Draw(vis)
            w_img, h_img = vis.size

            for o in objs: # Draw Predictions (Red)
                x0 = max(0.0, min(1.0, o["x_min"])) * w_img
                y0 = max(0.0, min(1.0, o["y_min"])) * h_img
                x1 = max(0.0, min(1.0, o["x_max"])) * w_img
                y1 = max(0.0, min(1.0, o["y_max"])) * h_img
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

            for bb in sample["boxes"]: # Draw Ground Truth (Green)
                x_min_n, y_min_n, width_n, height_n = bb.detach().cpu().tolist()
                x_min_px = x_min_n * w_img
                y_min_px = y_min_n * h_img
                x_max_px = (x_min_n + width_n) * w_img
                y_max_px = (y_min_n + height_n) * h_img
                x_min_px = max(0, min(w_img - 1, x_min_px))
                y_min_px = max(0, min(h_img - 1, y_min_px))
                x_max_px = max(0, min(w_img - 1, x_max_px))
                y_max_px = max(0, min(h_img - 1, y_max_px))
                draw.rectangle([x_min_px, y_min_px, x_max_px, y_max_px], outline="green", width=2)

            # Log to wandb (same logic, potentially update key)
            wandb.log({
                "eval/example_detect_replacement": # Consider a new key
                    wandb.Image(vis, caption="Detect via Inline Ref (Replacement Eval)")
            })

    # --- Return mAP (Same as original) ---
    # Add basic check for empty lists before calling compute_map
    if not preds or not gts:
        print("Warning: No predictions or ground truths to compute mAP.")
        return 0.0
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

        # convert pixel [xmin, ymin, xmax, ymax] to normalized [xmin, ymin, width, height]
        norm_boxes = []
        w_img, h_img = image.size
        for bbox in bboxes:
            x_min_px, y_min_px, x_max_px, y_max_px = bbox

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
        torch.set_default_device("cuda:1")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    wandb.init(
        project="moondream-ft",
        name="detect-ref-inline",
        config={
            "EPOCHS": EPOCHS,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
        },
    )

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(MODEL_PATH, model, device="cuda:1")

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
            # {"params": model.text.parameters()},
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
    mAP0 = eval_detect_inline(dataset, eval_idxs, model)
    wandb.log({"mAP_v1": mAP0})
    model.train()

    random.seed(3301)

    i = 0
    for epoch in range(EPOCHS):
        if epoch:
            random.shuffle(train_idxs)
            
        frac_class_epoch = 0.8 * (1 - epoch / (EPOCHS - 1))
        wandb.log({"frac_class_epoch": frac_class_epoch})
        
        for sample_idx in train_idxs:
            sample = dataset[sample_idx]
            i += 1

            torch.cuda.empty_cache()

            # 1) Compute vision embeddings for scene + reference
            img_emb = model._run_vision_encoder(sample["image"])      # [D]
            ref_emb = model._run_vision_encoder(sample["reference"])  # [D]

            with torch.no_grad():
                prefix_ids = model.config.tokenizer.templates["detect"]["prefix"]
                prefix_ids_w_leading_space = model.config.tokenizer.templates["detect"]["prefix"]+[220]
                suffix_ids = model.config.tokenizer.templates["detect"]["suffix"]
                prefix_emb = text_encoder(torch.tensor([prefix_ids], device=model.device), model.text).squeeze(0)
                prefix_emb_w_leading_space = text_encoder(torch.tensor([prefix_ids], device=model.device), model.text).squeeze(0)
                suffix_emb = text_encoder(torch.tensor([suffix_ids], device=model.device), model.text).squeeze(0)

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
            # so prefix_len = 1 (BOS) + 1 (IMG) + instr_len
            prefix_base = 1 + 1

            # Group ground‐truth boxes by “class” (we still use class_names, though
            # for DetectRef that token is just a placeholder)
            boxes_by_class = {}
            for box, cls in zip(sample["boxes"], sample["class_names"]):
                boxes_by_class.setdefault(cls, []).append(box)

            total_loss = 0.0

            # 3) For each “class”
            for _cls, boxes_list in boxes_by_class.items():
                use_class = (random.random() < frac_class_epoch)
                if use_class:
                    with torch.no_grad():
                        cls_emb = text_encoder(torch.tensor([
                            model.tokenizer.encode(
                                " "+_cls.replace('-', ' ').strip()
                            ).ids
                        ], device=model.device), model.text).squeeze(0)  
                    
                    instruction_emb = torch.cat([
                        prefix_emb.unsqueeze(0),   # [1, prefix_len, D]
                        cls_emb.unsqueeze(0),
                        ref_emb[None],             # [1,1,D]
                        suffix_emb.unsqueeze(0),   # [1, suffix_len, D]
                    ], dim=1)
                else:
                    instruction_emb = torch.cat([
                        prefix_emb_w_leading_space.unsqueeze(0),   # [1, prefix_len, D]
                        ref_emb[None],             # [1,1,D]
                        suffix_emb.unsqueeze(0),   # [1, suffix_len, D]
                    ], dim=1)

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

                # 4) Concatenate everything: [BOS] [IMG] [INST...] [CS...] [EOS]
                inputs_embeds = torch.cat([
                    bos_emb,           # [1,1,D]
                    img_emb[None],     # [1,1,D]
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
        mAP_e = eval_detect_inline(dataset, eval_idxs, model)
        wandb.log({"mAP_v1": mAP_e})
        model.train()

    wandb.finish()

    # Replace with your desired output location.
    save_file(
        model.state_dict(),
        "moondream_finetune_v2.safetensors",
    )


if __name__ == "__main__":
    """
    Replace paths with your appropriate paths.
    To run: python -m moondream.finetune.finetune_region

    1 epoch of fine-tuning on the example 'Waste Detection' dataset results in an
    increase in mAP from 61.82 to 69.82.
    """
    main()
