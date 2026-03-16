"""
    Adapted from https://github.com/konst-int-i/healnet by Marta Buetas-Arcas for multimodal survival datasets.

    Notes:
    - `tensors` is a list of length n_modalities. Elements can be None (modality absent for whole batch)
    or a tensor of shape (b, *spatial_dims, channels) before positional encoding / flattening.
    - `masks` is optional: list of boolean tensors matching each modality spatial shape (True where token exists).
    - `present` is optional: list of 1D boolean tensors of length b indicating whether a sample has that modality.
    If not provided, presence is derived from masks (if given) or from non-zero rows of tensors.
    - `mask` (singular) is legacy and will be used if `masks` is not provided.
"""

from math import pi, log
from functools import wraps
from typing import *

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce
import logging
logger = logging.getLogger(__name__)


class HealNet(nn.Module):
    def __init__(
        self,
        *,
        n_modalities: int,
        channel_dims: List,
        num_spatial_axes: List, 
        out_dims: int,
        depth: int = 3,
        num_freq_bands: int = 2,
        max_freq: float=10.,
        l_c: int = 128,
        l_d: int = 128,
        x_heads: int = 8,
        l_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        weight_tie_layers: bool = False,
        fourier_encode_data: bool = True,
        self_per_cross_attn: int = 1,
        final_classifier_head: bool = True,
        snn: bool = True,
    ):
        """
        Network architecture for easy-to-use multimodal fusion for any number and type of modalities.
        
        The input for each modality should be of shape ``(b, (*spatial_dims) c)``, where ``c`` corresponds to the dimensions 
        where positional encoding does not matter (e.g., color channels, set-based features, or tabular features). 

        Args:
            n_modalities (int): Maximum number of modalities for forward pass. Note that fewer modalities can be passed
                if modalities for individual samples are missing (see ``.forward()``)
            channel_dims (List[int]): Number of channels or tokens for each modality. Length must match ``n_modalities``. 
                The channel_dims are non-spatial dimensions where positional encoding is not required. 
            num_spatial_axes (List[int]): Spatial axes for each modality.The each spatial axis will be assigned positional 
                encodings, so that ``num_spatial_axis`` is 2 for 2D images, 3 for Video/3D images. 
            out_dims (int): Output shape of task-specific head. Forward pass returns logits of this shape. 
            num_freq_bands (int, optional): Number of frequency bands for positional encodings. Defaults to 2.
            max_freq (float, optional): Maximum frequency for positional encoding. Defaults to 10.
            l_c (int, optional): Number of channels for latent bottleneck array (akin to a "learned query array"). Defaults to 128.
            l_d (int, optional): Dimensions for latent bottleneck. Defaults to 128.
            x_heads (int, optional): Number of heads for cross attention. Defaults to 8.
            l_heads (int, optional): Number of heads for latent attention. Defaults to 8.
            cross_dim_head (int, optional): Dimension of each cross attention head. Defaults to 64.
            latent_dim_head (int, optional): Dimension of each latent attention head. Defaults to 64.
            attn_dropout (float, optional): Dropout rate for attention layers. Defaults to 0.
            ff_dropout (float, optional): Dropout rate for feed-forward layers. Defaults to 0.
            weight_tie_layers (bool, optional): False for weight sharing between fusion layers, True for specific 
                weights for each layer. Note that the number of parameters will multiply by ``depth`` if True. 
                Defaults to False.
            fourier_encode_data (bool, optional): Whether to use positional encoding. Recommended if meaningful spatial 
                spatial structure should be preserved. Defaults to True.
            self_per_cross_attn (int, optional): Number of self-attention layers per cross-attention layer. Defaults to 1.
            final_classifier_head (bool, optional): Whether to include a final classifier head. Defaults to True.
            snn (bool, optional): Whether to use a self-normalizing network. Defaults to True.

        Example: 
        
            ```python
            from healnet import HealNet
            from healnet.etl import MMDataset
            import torch
            import einops

            # synthetic data example
            n = 100 # number of samples
            b = 4 # batch size
            img_c = 3 # image channels
            tab_c = 1 # tabular channels
            tab_d = 2000 # tabular features
            # 2D dims
            h = 224 # image height
            w = 224 # image width
            # 3d dim
            d = 12

            tab_tensor = torch.rand(size=(n, tab_c, tab_d)) 
            img_tensor_2d = torch.rand(size=(n, h, w, img_c)) # h w c
            img_tensor_3d = torch.rand(size=(n, d, h, w, img_c)) # d h w c
            dataset = MMDataset([tab_tensor, img_tensor_2d, img_tensor_3d])

            [tab_sample, img_sample_2d, img_sample_3d] = dataset[0]

            # batch dim for illustration purposes
            tab_sample = einops.repeat(tab_sample, 'c d -> b c d', b=1) # spatial axis: None (pass as 1)
            img_sample_2d = einops.repeat(img_sample_2d, 'h w c -> b h w c', b=1) # spatial axes: h w
            img_sample_3d = einops.repeat(img_sample_3d, 'd h w c -> b d h w c', b=1) # spatial axes: d h w

            tensors = [tab_sample, img_sample_2d, img_sample_3d]


            model = HealNet(
                        n_modalities=3, 
                        channel_dims=[2000, 3, 3], # (2000, 3, 3) number of channels/tokens per modality
                        num_spatial_axes=[1, 2, 3], # (1, 2, 3) number of spatial axes (will be positionally encoded to preserve spatial information)
                        out_dims = 4
                    )

            # example forward pass
            logits = model(tensors)
            ```
        """
        
        
        super().__init__()
        assert len(channel_dims) == len(num_spatial_axes), 'input channels and input axis must be of the same length'
        assert len(num_spatial_axes) == n_modalities, 'input axis must be of the same length as the number of modalities'

        self.input_axes = num_spatial_axes
        self.input_channels=channel_dims
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.modalities = n_modalities
        self.self_per_cross_attn = self_per_cross_attn
        self.fourier_encode_data = fourier_encode_data

        # get fourier channels and input dims for each modality
        fourier_channels = []
        input_dims = []
        for axis in num_spatial_axes:
            fourier_channels.append((axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0)
        for f_channels, i_channels in zip(fourier_channels, channel_dims):
            input_dims.append(f_channels + i_channels)

        # initialise shared latent bottleneck
        self.latents = nn.Parameter(torch.randn(l_c, l_d))

        # modality-specific cross-attention layers
        funcs = []
        for m in range(n_modalities):
            funcs.append(lambda m=m: PreNorm(l_d, Attention(l_d, input_dims[m], heads = x_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dims[m]))
        cross_attn_funcs = tuple(map(cache_fn, tuple(funcs)))

        get_latent_attn = lambda: PreNorm(l_d, Attention(l_d, heads = l_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_cross_ff = lambda: PreNorm(l_d, FeedForward(l_d, dropout = ff_dropout, snn = snn))
        get_latent_ff = lambda: PreNorm(l_d, FeedForward(l_d, dropout = ff_dropout, snn = snn))

        get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(get_latent_attn(**cache_args, key = block_ind))
                self_attns.append(get_latent_ff(**cache_args, key = block_ind))


            cross_attn_layers = []
            for j in range(n_modalities):
                cross_attn_layers.append(cross_attn_funcs[j](**cache_args))
                cross_attn_layers.append(get_cross_ff(**cache_args))


            self.layers.append(nn.ModuleList(
                [*cross_attn_layers, self_attns])
            )

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(l_d),
            nn.Linear(l_d, out_dims)
        ) if final_classifier_head else nn.Identity()
    
    def forward(self,
                tensors: List[Union[torch.Tensor, None]],
                mask: Optional[torch.Tensor] = None,
                masks: Optional[List[Optional[torch.Tensor]]] = None,
                present: Optional[List[Optional[torch.Tensor]]] = None,
                return_embeddings: bool = False, 
                verbose: bool = True
                ):
        """
        Forward pass supporting per-sample missing modalities.
        - run attention ONLY for those samples with the modality present in that batch
        - For each modality i, compute present_idx = indices of samples that actually have modality i.
        - Call cross-attn only on that sliced sub-batch (x[present_idx], ctx[present_idx], mask_present).
        - Scatter the updated latents back to x[present_idx] = updated.
        - Samples without modality i are left unchanged for that modality (their latent stays as before).
        - Attention code remains unchanged.

        Args:
            tensors: list of length n_modalities. Each element either None or tensor (b, *spatial_dims, channels).
            mask: legacy single mask (applied to all modalities) OR None.
            masks: optional list of per-modality masks (same leading batch dim and spatial dims).
            present: optional list of 1D boolean tensors (shape (b,)) indicating sample-level presence for each modality.
            return_embeddings: if True, return the latent embeddings (b, n_latents, latent_dim) instead of logits.
            verbose: if True, logs debug messages.

        Returns:
            logits of shape (b, out_dims) or embeddings if return_embeddings.
        """
        if verbose:
            logger.debug("HealNet forward called. n_modalities=%d", self.modalities)

        missing_idx = [i for i, t in enumerate(tensors) if t is None]
        if verbose:
            logger.debug("Missing modalities indices: %s", missing_idx)

        # Determine batch size and a device/dtype to use:
        b = None
        global_device = None
        global_dtype = None
        
        # 1) Try find first non-None tensor
        for t in tensors:
            if t is not None:
                b = t.shape[0]
                global_device = t.device
                global_dtype = t.dtype
                break
        # 2) else try masks
        if b is None and masks is not None:
            for m in masks:
                if m is not None:
                    b = m.shape[0]
                    global_device = m.device
                    global_dtype = m.dtype
                    break
        # 3) else try present list
        if b is None and present is not None:
            for p in present:
                if p is not None:
                    b = int(p.numel())
                    global_device = p.device if isinstance(p, torch.Tensor) else None
                    global_dtype = torch.bool
                    break
        if b is None:
            raise ValueError("Cannot determine batch size: all modality tensors, masks and present lists are None. "
                            "Model requires at least one present modality or a provided mask/present vector.")

        preprocessed_tensors = [None] * len(tensors) # local list to store preprocessed tensors isntead of overwriting input

        # Preprocess modalities (fourier encoding, flatten)
        for i in range(len(tensors)):
            if i in missing_idx: 
                continue
            else: 
                data = tensors[i]
        
                if data is None:
                    continue
                if not isinstance(data, torch.Tensor):
                    data = torch.as_tensor(data)

                # get batch and spatial axes info
                try:
                    b_data, *axis, d_last = data.shape
                except Exception as e:
                    raise ValueError(f"Unexpected tensor shape for modality {i}: {getattr(data, 'shape', None)}") from e

                assert b_data == b, f"Batch size mismatch for modality {i}: tensor batch {b_data} != inferred batch {b}"
                device = data.device
                dtype = data.dtype

            # fourier encode for each modality
            if self.fourier_encode_data:
                axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
                pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
                enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
                enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
                enc_pos = repeat(enc_pos, '... -> b ...', b = b)
                data = torch.cat((data, enc_pos), dim = -1)
                

            # concat and flatten axis for each modality
            data = rearrange(data, 'b ... d -> b (...) d')
            preprocessed_tensors[i] = data

        tensors = preprocessed_tensors # use preprocessed tensors for forward pass

        # Create initial latents replicated to batch
        x = repeat(self.latents, 'n d -> b n d', b = b) # note: batch dim should be identical across modalities

        # Fusion loop with bucketed processing per modality
        for layer_idx, layer in enumerate(self.layers):
            for i in range(self.modalities):
                # skip completely-missing modalities (ORIGINAL missing modality handling)
                if i in missing_idx:
                    if verbose:
                        logger.debug("Skipping update in fusion layer %d for missing modality %d", layer_idx + 1, i + 1)
                    continue

                cross_attn = layer[i*2]
                cross_ff = layer[(i*2)+1]
                ctx = tensors[i]  # expected shape (b, n_ctx, d) when not None

                # Determine per-sample presence for this modality
                mod_present = None
                mod_mask = None

                # If user provided a present list entry for this modality, use it
                if isinstance(present, (list, tuple)) and len(present) > i and present[i] is not None:
                    # ensure bool tensor on correct device
                    mod_present = torch.as_tensor(present[i], dtype=torch.bool, device=x.device)
                    assert mod_present.ndim == 1 and mod_present.numel() == b, \
                        f"present[{i}] must be a 1D boolean tensor of length batch_size={b}"

                # Else if user provided a modality mask entry, derive presence from it
                elif isinstance(masks, (list, tuple)) and len(masks) > i and masks[i] is not None:
                    tmp_mask = rearrange(masks[i], 'b ... -> b (...)')  # flatten spatial dims
                    mod_mask = tmp_mask.to(device=x.device)
                    mod_present = mod_mask.any(dim = -1)

                # fallback: if no present/mask info, assume all samples present
                if mod_present is None:
                    mod_present = torch.ones(b, dtype=torch.bool, device=x.device)

                # present_idx: indices of samples that actually have modality i
                present_idx = mod_present.nonzero(as_tuple=True)[0]

                # If nobody in batch has this modality, skip updates for it
                if present_idx.numel() == 0:
                    if verbose:
                        logger.debug("No samples present for modality %d in layer %d. Skipping.", i, layer_idx)
                    continue

                # Fast-path: every sample present 
                if present_idx.numel() == b:
                    # sanity: context must not be None
                    if ctx is None:
                        # Provide an informative error instead of failing in LayerNorm with a cryptic message
                        raise RuntimeError(
                            f"Context tensor for modality {i} is None but present index indicates all samples present. "
                            "This indicates a mismatch between `tensors` returned by collate and the `present`/`masks` "
                            "passed into the model."
                        )
                    
                    # determine mask arg for cross_attn: prefer per-modality masks list if provided, else legacy mask
                    mask_arg = (masks[i] if isinstance(masks, (list,tuple)) and len(masks) > i else mask)
                    # Ensure mask_arg is on the same device as x (if not None)
                    if mask_arg is not None and isinstance(mask_arg, torch.Tensor):
                        mask_arg = mask_arg.to(device=x.device).to(dtype=torch.bool)
                        mask_flat = rearrange(mask_arg, 'b ... -> b (...)')
                        if mask_flat.shape[0] != b:
                            raise ValueError(f"Mask batch dimension mismatch for modality {i}: {mask_flat.shape[0]} != {b}")
                        if ctx.shape[1] != mask_flat.shape[1]:
                            raise ValueError(f"Mask token count mismatch for modality {i}: mask {mask_flat.shape[1]} != ctx {ctx.shape[1]}")


                    x = cross_attn(x, context = ctx, mask = mask_arg) + x
                    x = cross_ff(x) + x
                else:
                    # Bucketed path: slice only present samples, run cross-attn, scatter back
                    x_present = x[present_idx]                # (p, n_latents, d)
                    ctx_present = ctx[present_idx]           # (p, n_ctx, d)

                    mask_present = None
                    if mod_mask is not None:
                        mask_present = mod_mask[present_idx]
                        mask_present = mask_present.to(device=x.device)
                        # validate present mask shape
                        mask_pres_flat = rearrange(mask_present, 'b ... -> b (...)')
                        if mask_pres_flat.shape[0] != x_present.shape[0]:
                            raise ValueError(f"Present-mask batch mismatch for modality {i}: mask {mask_pres_flat.shape[0]} vs x_present {x_present.shape[0]}")
                        if ctx_present is not None and ctx_present.shape[1] != mask_pres_flat.shape[1]:
                            raise ValueError(f"Present-mask token mismatch for modality {i}: mask tokens {mask_pres_flat.shape[1]} != ctx_present tokens {ctx_present.shape[1]}")

                    updated = cross_attn(x_present, context = ctx_present, mask = mask_present) + x_present
                    updated = cross_ff(updated) + updated

                    # write updated latents back
                    x[present_idx] = updated

                # apply latent self-attention blocks for this layer 
                if self.self_per_cross_attn > 0:
                    self_attns = layer[-1]
                    # layer[-1] is a ModuleList of PreNorm blocks (alternating attention/ff)
                    for block in self_attns:
                        x = block(x) + x

        if return_embeddings:
            return x

        return self.to_logits(x)


    def get_attention_weights(self) -> List[torch.Tensor]:
        """
        Helper function which returns all attention weights for all attention layers in the model
        Returns:
            all_attn_weights: list of attention weights for each attention layer
        """
        all_attn_weights = []
        for module in self.modules():
            if isinstance(module, Attention):
                all_attn_weights.append(module.attn_weights)
        return all_attn_weights



# HELPERS/UTILS
"""
Helper class implementations based on: https://github.com/lucidrains/perceiver-pytorch
"""


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    """
    Fourier positional encoding for coordinate tensor x (..., n_dims).
    Returns tensor with sin/cos features concatenated and original coordinates appended.
    """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# ---------------------------
# Helper modules
# ---------------------------

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            if 'context' not in kwargs or kwargs['context'] is None:
                raise RuntimeError("PreNorm was constructed with context_dim but no context tensor was provided to the layer. "
                                   "Ensure cross-attention is called with context=<tensor> when the block expects a context.")
            
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class SELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.selu(gates)

class RELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.relu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., snn: bool = False):
        super().__init__()
        activation = SELU() if snn else GELU()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            activation,
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def temperature_softmax(logits, temperature=1.0, dim=-1):
    """
    Temperature scaled softmax
    Args:
        logits:
        temperature:
        dim:

    Returns:
    """
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=dim)



class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        # add leaky relu
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.LeakyReLU(negative_slope=1e-2)
        )

        self.attn_weights = None
        # self._init_weights()

    def _init_weights(self):
    # Use He initialization for Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # Initialize bias to zero if there's any
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        # attn = sim.softmax(dim = -1)
        attn = temperature_softmax(sim, temperature=0.5, dim=-1)
        self.attn_weights = attn
        attn = self.dropout(attn)


        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)