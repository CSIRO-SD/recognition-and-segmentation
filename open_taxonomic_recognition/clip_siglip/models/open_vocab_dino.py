"""Why DINO is not open-text by itself, and a minimal bridging pattern.

DINO/DINOv2/DINOv3-style models output image patches / CLS embeddings only — no paired text tower.
Open-vocabulary options:

1. **CLIP-style bridge (recommended for fair comparison)**  
   Freeze or partially freeze DINO image encoder; add a linear (or shallow MLP) projection
   W: R^{D_img} → R^{D_txt} matching a frozen **CLIP/SigLIP text encoder**.
   Train with image–caption pairs (scientific_name + locality + notes concatenation) contrastive vs text.

2. **Per-class textual prototypes**  
   Build string templates “a photo of {taxon_key}, …” encode with frozen CLIP text tower;
   DINO CLS → projector to CLIP vision dim; compare cosine to text prototypes — same trick as LiT / CLIP+DINO hybrids.

3. **Closed-set linear probe** only if you collapse train classes to softmax — not OV at test unseen taxa.

Implement (1)+(2) in `train/train_dino_projector.py` when you wire data; stub file documents the contract."""

from dataclasses import dataclass


@dataclass
class DINOBaselineNote:
    has_text_encoder: bool = False
    open_vocab_via: str = "external_clip_text_encoder_plus_learned_projection"
