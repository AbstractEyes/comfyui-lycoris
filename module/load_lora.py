import logging
import json
from . import adapters

from comfy import model_management  # ← Added for safe interrupt handling

import time
import re
from comfy.utils import ProgressBar

logger = logging.getLogger(__name__)

# Common LoRA CLIP key map used for model_lora_keys_clip
LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}

def load_lora(lora, to_load, log_missing=True, model_type_assumption="lokr"):
    """
    Custom load_lora that uses the enhanced adapter system.
    Args:
        lora: Dictionary containing the lora weights
        to_load: Dictionary mapping lora_key -> model_key
        log_missing: Whether to log keys that weren't loaded

    Returns:
        Dictionary of patches in format {model_key: patch_data}
        where patch_data is ("diff", (tensor,)) or ("dora", (diff, scale))
    """
    start_time = time.time()
    patch_dict = {}
    loaded_keys = set()

    # Metadata detection
    network_args = None
    ostris_mode = False

    if hasattr(lora, 'metadata') and isinstance(lora.metadata, dict):
        meta = lora.metadata
        algo = meta.get("ss_network_args", "")
        if isinstance(algo, str) and algo.strip().startswith("{"):
            try:
                network_args = json.loads(algo)
                logger.info(f"[load_lora] Network args: {network_args}")
            except json.JSONDecodeError as e:
                logger.warning(f"[load_lora] Failed to parse ss_network_args: {e}")
        if "software" in meta and "ai-toolkit" in str(meta.get("software")):
            ostris_mode = True
            logger.warning("[load_lora] Detected ai-toolkit model (OSTRIS). Enabling ostris_fallback mode.")

    # Patch: fallback for ostris-trained lycoris blocks
    if ostris_mode:
        for key in lora.keys():
            m = re.match(r"(lycoris_blocks_\d+_[a-z0-9_]+)\.lokr_w\d", key)
            if m:
                base = m.group(1)
                if base not in to_load:
                    model_key = f"diffusion_model.{base}.weight"
                    to_load[base] = model_key
                    logger.debug(f"[load_lora] Added fallback mapping: {base} -> {model_key}")

    logger.info(f"Starting load_lora with {len(to_load)} layers to process")
    pbar = ProgressBar(len(to_load))

    # Adapter loop
    for i, lora_key in enumerate(to_load):
        model_management.throw_exception_if_processing_interrupted()
        pbar.update_absolute(i)

        model_key = to_load[lora_key]

        alpha = lora.get(f"{lora_key}.alpha", None)
        if alpha is not None:
            alpha = alpha.item()
            loaded_keys.add(f"{lora_key}.alpha")

        dora_scale = lora.get(f"{lora_key}.dora_scale", None)
        if dora_scale is not None:
            loaded_keys.add(f"{lora_key}.dora_scale")

        # Adapter override detection
        adapter_result = None
        adapter_override = None
        if network_args:
            algo_name = network_args.get("algo", "").strip().lower()
            adapter_override = next((a for a in adapters if a.name == algo_name), None)

        if adapter_override:
            adapter_result = adapter_override.load(lora_key, lora, alpha, dora_scale, loaded_keys)
        else:
            for adapter_cls in adapters:
                if adapter_cls.is_applicable(lora_key, lora):
                    adapter_result = adapter_cls.load(lora_key, lora, alpha, dora_scale, loaded_keys)
                    if adapter_result:
                        break

        if adapter_result:
            patch_type, patch_data = adapter_result

            # Patch for Conv2d reshaping
            if patch_type == "diff" and len(patch_data) == 1:
                diff_tensor = patch_data[0]
                if model_key.endswith(".weight") and diff_tensor.dim() == 2:
                    if "skip_connection" in model_key or "op" in model_key:
                        out_ch = diff_tensor.shape[0]
                        in_ch = diff_tensor.shape[1]
                        diff_tensor = diff_tensor.reshape(out_ch, in_ch, 1, 1)
                    elif "in_layers.2" in model_key or "out_layers.3" in model_key:
                        out_ch = diff_tensor.shape[0]
                        in_ch = diff_tensor.shape[1]
                        kernel_size = 3
                        diff_tensor = diff_tensor.reshape(out_ch, in_ch, kernel_size, kernel_size)
                    adapter_result = (patch_type, (diff_tensor,))

            patch_dict[model_key] = adapter_result

        # Fallback: classic weight types
        for suffix in ("w_norm", "b_norm", "diff", "diff_b", "set_weight"):
            name = f"{lora_key}.{suffix}"
            val = lora.get(name, None)
            if val is not None:
                loaded_keys.add(name)
                if suffix in {"b_norm", "diff_b"}:
                    patch_dict[model_key.replace(".weight", ".bias")] = ("diff", (val,))
                elif suffix == "set_weight":
                    patch_dict[model_key] = ("set", (val,))
                else:
                    patch_dict[model_key] = ("diff", (val,))

    # Missing keys report
    if log_missing:
        for k in lora:
            if k not in loaded_keys:
                logger.warning(f"lora key not loaded: {k}")

    elapsed = time.time() - start_time
    logger.info(f"load_lora completed in {elapsed:.2f}s - {len(patch_dict)} patches created")
    return patch_dict


def model_lora_keys_clip(model, key_map={}):
    """
    Build CLIP model key mappings for LoRA loading.
    Maps from LoRA naming conventions to actual model weight names.
    """
    sdk = model.state_dict().keys()

    # text_encoders prefix handling
    for k in sdk:
        if k.endswith(".weight"):
            key_map[f"text_encoders.{k[:-len('.weight')]}"] = k

    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    clip_l_present, clip_g_present = False, False

    for b in range(32):  # CLIP layers
        for c in LORA_CLIP_MAP:
            h_key = f"clip_h.transformer.text_model.encoder.layers.{b}.{c}.weight"
            l_key = f"clip_l.transformer.text_model.encoder.layers.{b}.{c}.weight"
            g_key = f"clip_g.transformer.text_model.encoder.layers.{b}.{c}.weight"

            for name, flag, lora_te in [(h_key, None, "lora_te"), (l_key, 'l', "lora_te1"), (g_key, 'g', "lora_te2")]:
                if name in sdk:
                    mapped = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                    key_map[mapped] = name
                    key_map[f"{lora_te}_text_model_encoder_layers_{b}_{LORA_CLIP_MAP[c]}"] = name
                    key_map[f"text_encoder.text_model.encoder.layers.{b}.{c}"] = name
                    if flag == 'l':
                        clip_l_present = True
                    elif flag == 'g':
                        clip_g_present = True
                        if not clip_l_present:
                            key_map[f"lora_prior_te_text_model_encoder_layers_{b}_{LORA_CLIP_MAP[c]}"] = name
                            key_map[f"text_encoder_2.text_model.encoder.layers.{b}.{c}"] = name

    # Handle T5XXL transformer
    for k in sdk:
        if k.endswith(".weight"):
            if k.startswith("t5xxl.transformer."):
                l_key = k[len("t5xxl.transformer."):-len(".weight")]
                t5_index = 1 + int(clip_g_present) + int(clip_l_present)
                key_map[f"lora_te{t5_index}_{l_key.replace('.', '_')}"] = k
            elif k.startswith("hydit_clip.transformer.bert."):
                l_key = k[len("hydit_clip.transformer.bert."):-len(".weight")]
                key_map[f"lora_te1_{l_key.replace('.', '_')}"] = k

    # Handle projections
    for proj_key, lora_key in [("clip_g.transformer.text_projection.weight", "lora_te2_text_projection"),
                               ("clip_l.transformer.text_projection.weight", "lora_te1_text_projection")]:
        if proj_key in sdk:
            key_map[lora_key] = proj_key

    return key_map


def model_lora_keys_unet(model, key_map={}):
    """
    Build UNet model key mappings for LoRA loading.
    Maps from LoRA naming conventions to actual model weight names.
    """
    sdk = model.state_dict().keys()

    # Standard UNet keys
    for k in sdk:
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            base = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
            key_map[f"lora_unet_{base}"] = k
            key_map[k[:-len(".weight")]] = k

    # Import only what we need to avoid circular imports
    import comfy.utils
    import comfy.model_base

    # Get diffusers mapping if the model has a config
    if hasattr(model, 'model_config') and hasattr(model.model_config, 'unet_config'):
        diff_keys = comfy.utils.unet_to_diffusers(model.model_config.unet_config)
        for k, to in diff_keys.items():
            if k.endswith(".weight"):
                base = k[:-len(".weight")].replace(".", "_")
                unet_key = f"diffusion_model.{to}"
                for prefix in ("", "unet."):
                    key = f"{prefix}{k[:-len('.weight')].replace('.to_', '.processor.to_')}"
                    if key.endswith(".to_out.0"):
                        key = key[:-2]
                    key_map[key] = unet_key
                key_map[f"lora_unet_{base}"] = unet_key
                key_map[f"lycoris_{base}"] = unet_key

    # Handle model-specific special cases
    # We'll do basic type checking instead of importing all model types
    model_class_name = type(model).__name__

    if "StableCascade" in model_class_name:
        for k in sdk:
            if k.startswith("diffusion_model.") and k.endswith(".weight"):
                key = k[len("diffusion_model."):-len(".weight")]
                key_map[f"lora_prior_unet_{key}"] = k
    elif any(name in model_class_name for name in ["SD3", "Flux", "PixArt", "HunyuanDiT", "HiDream", "ACEStep"]):
        for k in sdk:
            if k.startswith("diffusion_model.") and k.endswith(".weight"):
                key = k[len("diffusion_model."):-len(".weight")]
                # Common transformer model patterns
                key_map[f"transformer.{key}"] = k
                key_map[f"base_model.model.{key}"] = k
                key_map[f"lora_transformer_{key.replace('.', '_')}"] = k
                key_map[f"lycoris_{key.replace('.', '_')}"] = k
                if "PixArt" in model_class_name:
                    key_map[f"unet.base_model.model.{key}"] = k

    return key_map