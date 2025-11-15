import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
from typing import Tuple, List


class ViTWithAttention(nn.Module):
    """
    A wrapper around torchvision ViT-B/16 that:
    - exposes CLS embedding
    - captures attention matrices from all transformer blocks
    - returns (logits, cls_embedding)
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        self.vit = vit_b_16(weights="DEFAULT" if pretrained else None)
        hidden_dim = self.vit.hidden_dim

        # Replace classification head
        self.vit.heads = nn.Linear(hidden_dim, num_classes)

        # Will store attention maps after forward pass
        self.attention_maps: List[torch.Tensor] = []

        # Register hooks to capture attention weights
        for idx, block in enumerate(self.vit.encoder.layers):
            block.attention.attn_drop.register_forward_hook(
                self._make_hook(idx)
            )

    # ----------------------------------------------------------------------
    def _make_hook(self, layer_idx: int):
        def hook(module, inp, out):
            # out: (B, heads, tokens, tokens)
            self.attention_maps.append(out.detach().cpu())
        return hook

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:         (B, num_classes)
            cls_embedding:  (B, hidden_dim)
        """
        self.attention_maps = []  # reset before each forward

        logits = self.vit(x)

        # Extract CLS token embedding
        tokens = self.vit._process_input(x)
        encoded = self.vit.encoder(tokens)
        cls_embedding = encoded[:, 0]

        return logits, cls_embedding

    # ----------------------------------------------------------------------
    def get_last_attention(self) -> torch.Tensor:
        if len(self.attention_maps) == 0:
            raise RuntimeError("Run forward() before accessing attention maps.")
        return self.attention_maps[-1]

    def get_all_attention(self) -> List[torch.Tensor]:
        return self.attention_maps
