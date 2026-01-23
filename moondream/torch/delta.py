"""
Delta-Vector VLM Extension for Moondream.

This module implements the "Residual VLM" architecture that enables efficient
multi-turn visual processing by compressing subsequent images into delta vectors.

Architecture:
- First image (I-Frame): Full 729 tokens via standard vision encoder
- Subsequent images (P-Frames): 64 compressed delta tokens via DeltaProjector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple
from PIL import Image

from .vision import vision_encoder, vision_projection, prepare_crops
from .image_crops import reconstruct_from_crops
from .text import text_encoder, text_decoder, lm_head
from .config import MoondreamConfig


class DeltaProjector(nn.Module):
    """
    Compresses the difference between two 27x27 feature grids into a smaller sequence.
    
    Takes two sets of vision features (current and previous frame) and outputs
    a compressed delta representation suitable for injection into the text decoder.
    
    Args:
        vision_dim: Dimension of the vision encoder output (default: 2048)
        out_dim: Dimension expected by the text model (default: 2048)
        grid_size: Spatial grid size of vision features (default: 27, from 378/14)
        out_tokens: Number of output tokens (default: 64, i.e., 8x8 grid)
    """
    
    def __init__(
        self,
        vision_dim: int = 2048,
        out_dim: int = 2048,
        grid_size: int = 27,
        out_tokens: int = 64,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.out_tokens = out_tokens
        
        # Calculate target pooling size (e.g., 64 tokens -> 8x8 grid)
        self.pool_size = int(out_tokens ** 0.5)
        assert self.pool_size ** 2 == out_tokens, "out_tokens must be a perfect square"
        
        # 1. Spatial Downsampling: 27x27 -> 8x8
        self.pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        
        # 2. Semantic Projection MLP
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, vision_dim * 2),
            nn.GELU(),
            nn.Linear(vision_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )
        
        # 3. Learned positional embeddings for delta tokens
        self.pos_emb = nn.Parameter(torch.randn(1, out_tokens, out_dim) * 0.02)
    
    def forward(
        self, 
        curr_features: torch.Tensor, 
        prev_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute compressed delta between current and previous frame features.
        
        Args:
            curr_features: Current frame features [B, 729, vision_dim]
            prev_features: Previous frame features [B, 729, vision_dim]
            
        Returns:
            Delta embeddings [B, out_tokens, out_dim]
        """
        # 1. Compute raw delta in latent space
        diff = curr_features - prev_features
        
        # 2. Reshape to spatial grid [B, vision_dim, 27, 27]
        B, N, C = diff.shape
        assert N == self.grid_size ** 2, f"Expected {self.grid_size**2} tokens, got {N}"
        diff = diff.transpose(1, 2).view(B, C, self.grid_size, self.grid_size)
        
        # 3. Pool to target size [B, vision_dim, pool_size, pool_size]
        pooled = self.pool(diff)
        
        # 4. Flatten to sequence [B, out_tokens, vision_dim]
        flat = pooled.flatten(2).transpose(1, 2)
        
        # 5. Project and add positional embeddings
        projected = self.proj(flat)
        return projected + self.pos_emb


class ResidualMoondream(nn.Module):
    """
    Wrapper around MoondreamModel that adds delta-vector capabilities.
    
    The base model is frozen; only the DeltaProjector is trainable.
    Supports both inference (using KV-cache) and training forward passes.
    
    Args:
        base_model: The underlying MoondreamModel instance
        out_tokens: Number of delta tokens (default: 64)
    """
    
    def __init__(self, base_model: nn.Module, out_tokens: int = 64):
        super().__init__()
        self.model = base_model
        self.config = base_model.config
        self.out_tokens = out_tokens
        
        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Add trainable delta projector
        self.delta_proj = DeltaProjector(
            vision_dim=self.config.vision.proj_out_dim,
            out_dim=self.config.text.dim,
            out_tokens=out_tokens,
        )
    
    @property
    def device(self):
        return self.model.device
    
    @property
    def dtype(self):
        return self.model.vision.pos_emb.dtype
    
    def encode_image_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of image tensors to vision features.
        
        This bypasses tiling and assumes images are already preprocessed
        to 378x378. Used for training with batched tensors.
        
        Args:
            images: Preprocessed image tensors [B, 3, 378, 378]
            
        Returns:
            Vision features [B, 729, proj_out_dim]
        """
        with torch.no_grad():
            # Run vision encoder (single tile, no reconstruction needed)
            outputs = self.model._vis_enc(images)  # [B, 729, enc_dim]
            
            # For single-tile images, global_features = outputs (no local crops)
            # We need to project through vision_projection
            # vision_projection expects: global_features [729, enc_dim] and 
            # reconstructed [H, W, enc_dim]
            
            B = images.shape[0]
            all_features = []
            
            for i in range(B):
                global_feat = outputs[i]  # [729, enc_dim]
                
                # For single tile, reconstructed is just the reshaped global features
                # Reshape to spatial for projection
                enc_n_layers = self.config.vision.enc_n_layers  # 27
                enc_dim = self.config.vision.enc_dim
                
                reconstructed = global_feat.view(enc_n_layers, enc_n_layers, enc_dim)
                
                # Run vision projection
                projected = vision_projection(
                    global_feat, 
                    reconstructed, 
                    self.model.vision, 
                    self.config.vision
                )  # [729, proj_out_dim]
                
                all_features.append(projected)
            
            return torch.stack(all_features, dim=0)  # [B, 729, proj_out_dim]
    
    def compute_delta(
        self,
        curr_features: torch.Tensor,
        prev_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute delta embeddings between current and previous frame.
        
        This is the only part with gradients during training.
        
        Args:
            curr_features: Current frame features [B, 729, dim]
            prev_features: Previous frame features [B, 729, dim]
            
        Returns:
            Delta embeddings [B, out_tokens, dim]
        """
        return self.delta_proj(curr_features, prev_features)
    
    def training_forward(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
        text_tokens: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass with proper gradient flow.
        
        Uses KV-cache prefill continuation approach:
        1. Prefill with I-Frame features (frozen)
        2. Continue prefill with delta embeddings (trainable)
        3. Continue with text tokens and compute loss
        
        Args:
            img_prev: Previous frame images [B, 3, 378, 378]
            img_curr: Current frame images [B, 3, 378, 378]
            text_tokens: Text token IDs [B, seq_len]
            labels: Label token IDs for loss [B, seq_len] (-100 for ignored)
            
        Returns:
            Tuple of (loss, logits)
        """
        B = img_prev.shape[0]
        device = img_prev.device
        
        # 1. Extract frozen vision features
        with torch.no_grad():
            feat_prev = self.encode_image_tensor(img_prev)  # [B, 729, dim]
            feat_curr = self.encode_image_tensor(img_curr)  # [B, 729, dim]
        
        # 2. Compute trainable delta (this has gradients!)
        delta_emb = self.compute_delta(feat_curr, feat_prev)  # [B, 64, dim]
        
        # 3. Get text embeddings (frozen)
        with torch.no_grad():
            text_emb = text_encoder(text_tokens, self.model.text)  # [B, seq_len, dim]
            
            # BOS embedding
            bos_tokens = torch.tensor(
                [[self.config.tokenizer.bos_id]], 
                device=device
            ).expand(B, -1)
            bos_emb = text_encoder(bos_tokens, self.model.text)  # [B, 1, dim]
        
        # 4. Concatenate all embeddings
        # Sequence: [BOS] + [I-Frame (729)] + [Delta (64)] + [Text]
        inputs_embeds = torch.cat([
            bos_emb,           # [B, 1, dim]
            feat_prev,         # [B, 729, dim] 
            delta_emb,         # [B, 64, dim]
            text_emb,          # [B, seq_len, dim]
        ], dim=1)
        
        seq_len = inputs_embeds.shape[1]
        
        # 5. Create causal attention mask
        attn_mask = self.model.attn_mask[:, :, :seq_len, :]
        
        # 6. Create position IDs
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        
        # 7. Run through text decoder
        hidden_states = text_decoder(
            inputs_embeds,
            self.model.text,
            attn_mask,
            pos_ids,
            self.config.text,
            lora=None,
        )  # [B, seq_len, dim]
        
        # 8. Compute logits for the text portion only
        # The text tokens start at position: 1 + 729 + 64 = 794
        text_start = 1 + 729 + self.out_tokens
        text_hidden = hidden_states[:, text_start:, :]
        
        # Simple lm_head computation (not using the built-in which takes last only)
        from .layers import layer_norm
        text_hidden_normed = layer_norm(text_hidden, self.model.text.post_ln)
        logits = text_hidden_normed @ self.model.text.lm_head.weight.T
        if hasattr(self.model.text.lm_head, 'bias') and self.model.text.lm_head.bias is not None:
            logits = logits + self.model.text.lm_head.bias
        
        # 9. Compute cross-entropy loss
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        return loss, logits
    
    def inference_with_delta(
        self,
        image_prev: Image.Image,
        image_curr: Image.Image,
        question: str,
        **kwargs,
    ) -> dict:
        """
        Run inference with delta-vector for two consecutive frames.
        
        Args:
            image_prev: Previous frame (PIL Image)
            image_curr: Current frame (PIL Image) 
            question: Question to answer about the difference
            **kwargs: Additional arguments passed to model.query()
            
        Returns:
            Dict with 'answer' key containing the response
        """
        # 1. Encode previous image (I-Frame) - populates KV cache
        encoded_prev = self.model.encode_image(image_prev)
        
        # 2. Get features for both images to compute delta
        with torch.no_grad():
            feat_prev = self.model._run_vision_encoder(image_prev)  # [729, dim]
            feat_curr = self.model._run_vision_encoder(image_curr)  # [729, dim]
        
        # 3. Compute delta embedding
        delta_emb = self.delta_proj(
            feat_curr.unsqueeze(0), 
            feat_prev.unsqueeze(0)
        )  # [1, 64, dim]
        
        # 4. Continue prefill with delta tokens
        pos = encoded_prev.pos
        delta_mask = self.model.attn_mask[:, :, pos:pos + self.out_tokens, :]
        delta_pos_ids = torch.arange(
            pos, pos + self.out_tokens, 
            dtype=torch.long, 
            device=self.device
        )
        
        with torch.inference_mode():
            self.model._prefill(delta_emb, delta_mask, delta_pos_ids, lora=None)
        
        new_pos = pos + self.out_tokens
        
        # 5. Load the KV cache (it was already updated by _prefill)
        # and continue with query
        # Create a fake EncodedImage with updated position
        from .moondream import EncodedImage
        encoded_with_delta = EncodedImage(
            pos=new_pos,
            caches=[
                (
                    b.kv_cache.k_cache[:, :, :new_pos, :].clone(),
                    b.kv_cache.v_cache[:, :, :new_pos, :].clone(),
                )
                for b in self.model.text.blocks
            ],
        )
        
        # 6. Run the query with the delta-enhanced context
        self.model.load_encoded_image(encoded_with_delta)
        
        # Build prompt tokens
        prompt_toks = self.config.tokenizer.templates["query"]["prefix"]
        question_toks = self.model.tokenizer.encode(question).ids
        prompt_tokens = torch.tensor(
            [prompt_toks + question_toks + [self.config.tokenizer.thinking_id]],
            device=self.device,
        )
        
        # Generate response
        result_tokens = []
        for token in self.model._generate_answer(
            prompt_tokens, 
            new_pos, 
            eos_id=self.config.tokenizer.eos_id,
        ):
            result_tokens.append(token)
        
        return {"answer": "".join(result_tokens)}
