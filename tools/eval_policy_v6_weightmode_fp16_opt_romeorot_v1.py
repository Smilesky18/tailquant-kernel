#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OPT FP16 GPTQ PPL eval with RoMeO-style split rotation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for item in (TOOLS, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", ROOT / "kernel_quant/scripts"):
    text = str(item)
    if text not in sys.path:
        sys.path.insert(0, text)

import eval_policy_v6_weightmode_v1 as BASE  # noqa: E402
import opt_romeorot_compat_v1 as OPTC  # noqa: E402


H = BASE.H
NORM_SCALE_EPS = float(os.environ.get("ROMEO_ROT_NORM_SCALE_EPS", "1e-3"))


def cast_fp16_and_alias_layers(model):
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point() and p.dtype != torch.float16:
                p.data = p.data.to(dtype=torch.float16)
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        model.model.layers = model.model.decoder.layers
    return model


_GET_MODEL = H.model_utils.get_model
_COLLECT = H.collect_target_modules
_CFG_KEY = H.module_name_to_cfg_key
_LAYER_ID = H.get_layer_id_from_module_name
_OFFLINE = H.apply_offline_weight_rotation
_ONLINE = H.register_online_rotation_hooks
_ADD_ACTQUANT = H.quant_utils.add_actquant
_GPTQ_FWRD = H.gptq_utils.gptq_fwrd


def get_model(*args, **kwargs):
    return cast_fp16_and_alias_layers(_GET_MODEL(*args, **kwargs))




def _slice_opt_attention_mask(mask, seq_len):
    if mask is None:
        return None
    if mask.dim() == 2:
        return mask[:, -seq_len:]
    if mask.dim() == 4:
        return mask[:, :, -seq_len:, -seq_len:]
    return mask


def _virtual_module_for_gptq(layer_id, name):
    clean = name.replace(".module", "")
    if clean == "self_attn.out_proj":
        clean = "self_attn.o_proj"
    elif clean == "fc1":
        clean = "mlp.up_proj"
    elif clean == "fc2":
        clean = "mlp.down_proj"
    return f"model.layers.{layer_id}.{clean}"


@torch.no_grad()
def gptq_fwrd_opt(model, dataloader, dev, args):
    q = H.quant_utils
    g = H.gptq_utils
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if getattr(model.model.decoder, "final_layer_norm", None) is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def __getattr__(self, name):
            if name == "module":
                return super().__getattr__(name)
            return getattr(self.module, name)
        def forward(self, inp, **kwargs):
            if cache["i"] >= args.nsamples:
                raise ValueError
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        if cache["i"] >= args.nsamples:
            break
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    if cache["i"] != args.nsamples:
        raise RuntimeError(f"OPT GPTQ captured {cache['i']} samples, expected {args.nsamples}")

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if getattr(model.model.decoder, "final_layer_norm", None) is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)

    quantizers = {}
    sequential = [
        ["self_attn.k_proj.module", "self_attn.v_proj.module", "self_attn.q_proj.module"],
        ["self_attn.out_proj.module"],
        ["fc1.module"],
        ["fc2.module"],
    ]
    for i in range(len(layers)):
        print()
        print(f"Layer {i}:", flush=True, end=" " )
        layer = layers[i].to(dev)
        full = q.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            if not subset:
                continue
            gptq = {}
            for name in subset:
                print(name, end="  ", flush=True)
                gptq[name] = g.GPTQ(subset[name])
                gptq[name].args = args
                gptq[name].quantizer = q.WeightQuantizer()
                gptq[name].quantizer.configure(args.w_bits, perchannel=True, sym=not args.w_asym, mse=args.w_clip)

            def add_batch(local_name):
                def hook(_, inp, out):
                    gptq[local_name].add_batch(inp[0].data, out.data)
                return hook

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
            for j in range(args.nsamples):
                layer_kwargs = {"attention_mask": _slice_opt_attention_mask(attention_mask, int(inps[j].shape[0]))}
                outs[j] = layer(inps[j].unsqueeze(0), **{k:v for k,v in layer_kwargs.items() if v is not None})[0]
            for h in handles:
                h.remove()

            for name in subset:
                prev = getattr(args, "weight_scale_current_module", None)
                args.weight_scale_current_module = _virtual_module_for_gptq(i, name)
                try:
                    gptq[name].fasterquant(
                        percdamp=args.percdamp, groupsize=args.w_groupsize, actorder=args.act_order, static_groups=False
                    )
                finally:
                    if prev is None:
                        if hasattr(args, "weight_scale_current_module"):
                            delattr(args, "weight_scale_current_module")
                    else:
                        args.weight_scale_current_module = prev
                quantizers[_virtual_module_for_gptq(i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            layer_kwargs = {"attention_mask": _slice_opt_attention_mask(attention_mask, int(inps[j].shape[0]))}
            outs[j] = layer(inps[j].unsqueeze(0), **{k:v for k,v in layer_kwargs.items() if v is not None})[0]
        layers[i] = layer.cpu()
        del layer
        if "gptq" in locals():
            del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    H.utils.cleanup_memory(verbos=True)
    print()
    print("[OPT GPTQ] done", flush=True)
    return quantizers


def safe_add_actquant(module, name=""):
    q = H.quant_utils
    if isinstance(module, q.ActQuantWrapper):
        return
    for child_name, child in list(module._modules.items()):
        if isinstance(child, q.ActQuantWrapper):
            continue
        if isinstance(child, torch.nn.Linear):
            module._modules[child_name] = q.ActQuantWrapper(child)
        elif isinstance(child, torch.nn.Sequential):
            replaced = []
            for sub in child:
                if isinstance(sub, q.ActQuantWrapper):
                    replaced.append(sub)
                elif isinstance(sub, torch.nn.Linear):
                    replaced.append(q.ActQuantWrapper(sub))
                else:
                    replaced.append(sub)
            module._modules[child_name] = torch.nn.Sequential(*replaced)
        elif isinstance(child, torch.nn.ModuleList):
            replaced = []
            for sub in child:
                if isinstance(sub, q.ActQuantWrapper):
                    replaced.append(sub)
                elif isinstance(sub, torch.nn.Linear):
                    replaced.append(q.ActQuantWrapper(sub))
                else:
                    replaced.append(sub)
            module._modules[child_name] = torch.nn.ModuleList(replaced)
    for child_name, child in module.named_children():
        safe_add_actquant(child, f"{name}.{child_name}" if name else child_name)


def main():
    H.model_utils.get_model = get_model
    H.collect_target_modules = OPTC.opt_collect_target_modules
    H.module_name_to_cfg_key = OPTC.opt_module_name_to_cfg_key
    H.get_layer_id_from_module_name = OPTC.opt_get_layer_id_from_module_name
    H.apply_offline_weight_rotation = lambda model, rot_flags: OPTC.apply_romeo_offline_weight_rotation_opt(
        model, H, rot_flags, NORM_SCALE_EPS
    )
    H.register_online_rotation_hooks = lambda model, rot_flags, had_cache: OPTC.register_romeo_online_rotation_hooks_opt(
        model, H, rot_flags, had_cache, NORM_SCALE_EPS
    )
    H.quant_utils.add_actquant = safe_add_actquant
    H.gptq_utils.gptq_fwrd = gptq_fwrd_opt
    try:
        BASE.main()
    finally:
        H.model_utils.get_model = _GET_MODEL
        H.collect_target_modules = _COLLECT
        H.module_name_to_cfg_key = _CFG_KEY
        H.get_layer_id_from_module_name = _LAYER_ID
        H.apply_offline_weight_rotation = _OFFLINE
        H.register_online_rotation_hooks = _ONLINE
        H.quant_utils.add_actquant = _ADD_ACTQUANT
        H.gptq_utils.gptq_fwrd = _GPTQ_FWRD


if __name__ == "__main__":
    main()
