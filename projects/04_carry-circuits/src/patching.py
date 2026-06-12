# patching utilities for carry circuit discovery
import torch
import numpy as np

from config import N_LAYERS, N_HEADS
from counterfactual import ANSWER_LOGIT_POS, ANSWER_SEQ_POS


def compute_logit_diff(logits, clean_tok, corrupted_tok, logit_pos):
    # logit[clean_answer] - logit[corrupted_answer] at logit_pos
    pos_logits = logits[:, logit_pos, :]
    cl = pos_logits.gather(1, clean_tok.unsqueeze(1)).squeeze(1)
    co = pos_logits.gather(1, corrupted_tok.unsqueeze(1)).squeeze(1)
    return cl - co

def compute_recovery(clean_ld, patched_ld, corrupted_ld):
    # fraction of clean-corrupted gap recovered
    denom = clean_ld - corrupted_ld
    safe = denom.abs() > 0.01
    result = torch.zeros_like(denom)
    result[safe] = (patched_ld[safe] - corrupted_ld[safe]) / denom[safe]
    return result


def compute_ie_stolfo(clean_ld, patched_ld, corrupted_ld):
    # stolfo symmetric IE with clamped denominator
    denom = clean_ld - corrupted_ld
    denom = torch.where(denom.abs() < 1e-10, torch.ones_like(denom), denom)
    return (patched_ld - corrupted_ld) / denom

def _resid_hook(layer):
    if layer < N_LAYERS:
        return f"blocks.{layer}.hook_resid_pre"
    return f"blocks.{N_LAYERS - 1}.hook_resid_post"

def _component_hook(comp, layer):
    return f"blocks.{layer}.hook_{comp}_out"

def _head_hook(layer):
    return f"blocks.{layer}.attn.hook_result"

def _pattern_hook(layer):
    return f"blocks.{layer}.attn.hook_pattern"


def patch_residual_stream(model, clean_input, corrupted_input, layer, pos):
    # denoising: run corrupted, patch in clean resid_pre at (layer, pos)
    _, clean_cache = model.run_with_cache(clean_input)
    hook = _resid_hook(layer)
    clean_act = clean_cache[hook].clone()

    def hook_fn(activation, **kwargs):
        activation[:, pos, :] = clean_act[:, pos, :]
        return activation

    return model.run_with_hooks(corrupted_input, fwd_hooks=[(hook, hook_fn)])

def patch_component(model, clean_input, corrupted_input, component_type, layer, pos=None):
    _, clean_cache = model.run_with_cache(clean_input)
    hook = _component_hook(component_type, layer)
    clean_act = clean_cache[hook].clone()

    def hook_fn(activation, **kwargs):
        if pos is not None:
            activation[:, pos, :] = clean_act[:, pos, :]
        else:
            activation[:] = clean_act
        return activation

    return model.run_with_hooks(corrupted_input, fwd_hooks=[(hook, hook_fn)])


def patch_head(model, clean_input, corrupted_input, layer, head, pos=None):
    # patch single attention head output
    _, clean_cache = model.run_with_cache(clean_input)
    hook = _head_hook(layer)
    clean_act = clean_cache[hook].clone()

    def hook_fn(activation, **kwargs):
        if pos is not None:
            activation[:, pos, head, :] = clean_act[:, pos, head, :]
        else:
            activation[:, :, head, :] = clean_act[:, :, head, :]
        return activation

    return model.run_with_hooks(corrupted_input, fwd_hooks=[(hook, hook_fn)])

# print("patching loaded")  # debug

def _pairs_to_tensors(pairs, start, end):
    batch = pairs[start:end]
    cl = torch.tensor([p["clean_tokens"] for p in batch], dtype=torch.long)
    co = torch.tensor([p["corrupted_tokens"] for p in batch], dtype=torch.long)
    return cl[:, :-1], co[:, :-1], cl, co


def compute_baselines(model, pairs, batch_size=64):
    n = len(pairs)
    ld = {"clean": {i: [] for i in range(4)}, "corrupted": {i: [] for i in range(4)}}

    with torch.no_grad():
        for s in range(0, n, batch_size):
            cl_in, co_in, cl_full, co_full = _pairs_to_tensors(pairs, s, s + batch_size)
            cl_logits = model(cl_in)
            co_logits = model(co_in)

            for ai in range(4):
                ct = cl_full[:, ANSWER_SEQ_POS[ai]]
                rt = co_full[:, ANSWER_SEQ_POS[ai]]
                lp = ANSWER_LOGIT_POS[ai]
                ld["clean"][ai].append(compute_logit_diff(cl_logits, ct, rt, lp))
                ld["corrupted"][ai].append(compute_logit_diff(co_logits, ct, rt, lp))

    return {
        ai: {"clean": torch.cat(ld["clean"][ai]),
             "corrupted": torch.cat(ld["corrupted"][ai])}
        for ai in range(4)
    }

def run_patching(model, pairs, patch_fn, baselines, batch_size=64, **patch_kwargs):
    n = len(pairs)
    patched_lds = {i: [] for i in range(4)}

    with torch.no_grad():
        for s in range(0, n, batch_size):
            cl_in, co_in, cl_full, co_full = _pairs_to_tensors(pairs, s, s + batch_size)
            patched_logits = patch_fn(model, cl_in, co_in, **patch_kwargs)

            for ai in range(4):
                ct = cl_full[:, ANSWER_SEQ_POS[ai]]
                rt = co_full[:, ANSWER_SEQ_POS[ai]]
                lp = ANSWER_LOGIT_POS[ai]
                patched_lds[ai].append(compute_logit_diff(patched_logits, ct, rt, lp))

    results = {}
    for ai in range(4):
        pl = torch.cat(patched_lds[ai])
        cl = baselines[ai]["clean"]
        co = baselines[ai]["corrupted"]
        rec = compute_recovery(cl, pl, co)
        results[ai] = {
            "recovery_mean": rec.mean().item(),
            "recovery_se": (rec.std().item() / max(len(rec) ** 0.5, 1)),
            "patched_ld_mean": pl.mean().item(),
            "clean_ld_mean": cl.mean().item(),
            "corrupted_ld_mean": co.mean().item(),
        }
    return results
