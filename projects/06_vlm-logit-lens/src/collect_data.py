"""Self-contained logit-lens sweep for Qwen2-VL-2B-Instruct on a synthetic
1250-image dataset (color, shape, count, spatial, binding). Writes
logit_lens_results.csv. Designed for Colab T4; runs on CPU/MPS too, slowly.

  pip install "transformers>=4.52" torch accelerate pillow qwen-vl-utils tqdm pandas

Env knobs: HF_HUB_DISABLE_XET=1 (download stalls), MAX_IMAGES=N (smoke test).
"""
import os, csv, json, math, random, time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

SEED = 42
random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
print(f"[init] device={DEVICE}  dtype={DTYPE}")

# Single-token surface forms for every answer word in the Qwen2 tokenizer.
TARGET_TOKEN_IDS = {
    "red":      {1151, 2518, 3731, 6033},
    "blue":     {6303, 8697, 10331, 12203},
    "green":    {6176, 7840, 13250, 19576},
    "yellow":   {13753, 25462, 27869, 47699},
    "circle":   {12671, 21224, 25199, 25857},
    "square":   {9334, 15619, 33271, 37476},
    "triangle": {21495, 38031, 51942, 55114},
    "star":     {6774, 7679, 11870, 12699},
    "left":     {2115, 2359, 5415, 13727},
    "right":    {1290, 1291, 5979, 10083},
    "above":    {3403, 43610, 48432, 58807},
    "below":    {3685, 21193, 38214, 53177},
    "1": {16}, "2": {17}, "3": {18}, "4": {19}, "5": {20},
    "one":   {603, 825, 3776, 3966},
    "two":   {1378, 9043, 11613, 19789},
    "three": {2326, 14513, 19641, 27856},
    "four":  {3040, 13322, 26972, 34024},
    "five":  {4236, 20924, 37020, 52670},
}

COUNT_LABEL_TO_WORD = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five"}

PROMPTS = {
    "color": {
        "question": "What color is the object? Answer with one word.",
        "prefill":  "The color is",
        "targets":  ["red", "blue", "green", "yellow"],
    },
    "shape": {
        "question": "What shape is the object? Answer with one word.",
        "prefill":  "The shape is a",
        "targets":  ["circle", "square", "triangle", "star"],
    },
    "count": {
        "question": "How many objects are in the image? Reply with a single number word only (one, two, three, four, or five).",
        "prefill":  "There are",
        "targets":  ["1", "2", "3", "4", "5"],
    },
    "spatial": {
        "question": "Is the red circle above, below, left of, or right of the blue square? Answer with one word.",
        "prefill":  "Answer:",
        "targets":  ["left", "right", "above", "below"],
    },
    "binding": {
        "question": "What color is the circle? Answer with one word.",
        "prefill":  "The circle is",
        "targets":  ["red", "blue"],
    },
}

CATEGORY_CLASSES = {
    "color":   ["red", "blue", "green", "yellow"],
    "shape":   ["circle", "square", "triangle", "star"],
    "count":   ["one", "two", "three", "four", "five"],
    "spatial": ["left", "right", "above", "below"],
    "binding": ["red", "blue"],
}

NUM_LAYERS = 28
IMAGE_SIZE = 448
BG = (208, 208, 208)
COLOR_RGB = {
    "red":    (255, 0, 0),
    "blue":   (0, 0, 255),
    "green":  (0, 200, 0),
    "yellow": (255, 215, 0),
}


def _new_canvas():
    return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BG)


def _draw_circle(d, cx, cy, r, fill, outline=None, width=1):
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=width)


def _draw_square(d, cx, cy, side, fill, outline=None, width=1):
    h = side / 2
    d.rectangle((cx - h, cy - h, cx + h, cy + h), fill=fill, outline=outline, width=width)


def _draw_triangle(d, cx, cy, side, fill, outline=None, width=1):
    h = side * math.sqrt(3) / 2
    pts = [(cx, cy - 2 * h / 3), (cx - side / 2, cy + h / 3), (cx + side / 2, cy + h / 3)]
    d.polygon(pts, fill=fill, outline=outline, width=width)


def _draw_star(d, cx, cy, r_outer, fill, outline=None, width=1, points=5, r_inner=25):
    pts = []
    for i in range(points * 2):
        ang = -math.pi / 2 + i * math.pi / points
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    d.polygon(pts, fill=fill, outline=outline, width=width)


def gen_color_set():
    items = []
    for color in PROMPTS["color"]["targets"]:
        for idx in range(50):
            random.seed(f"42-color-{color}-{idx}")
            jx = random.randint(-30, 30); jy = random.randint(-30, 30)
            cx, cy = IMAGE_SIZE // 2 + jx, IMAGE_SIZE // 2 + jy
            img = _new_canvas()
            _draw_circle(ImageDraw.Draw(img), cx, cy, 60, fill=COLOR_RGB[color])
            items.append({
                "category": "color", "gt_label": color, "image": img,
                "filename": f"color/{color}_{idx:03d}.png",
                "pair_id": None, "pair_side": None,
            })
    return items


def gen_shape_set():
    items = []
    fill = (255, 255, 255); outline = (0, 0, 0); lw = 3
    for shape in PROMPTS["shape"]["targets"]:
        for idx in range(50):
            random.seed(f"42-shape-{shape}-{idx}")
            jx = random.randint(-30, 30); jy = random.randint(-30, 30)
            cx, cy = IMAGE_SIZE // 2 + jx, IMAGE_SIZE // 2 + jy
            img = _new_canvas(); d = ImageDraw.Draw(img)
            if shape == "circle":     _draw_circle(d, cx, cy, 60, fill, outline, lw)
            elif shape == "square":   _draw_square(d, cx, cy, 120, fill, outline, lw)
            elif shape == "triangle": _draw_triangle(d, cx, cy, 120, fill, outline, lw)
            elif shape == "star":     _draw_star(d, cx, cy, 60, fill, outline, lw, r_inner=25)
            items.append({
                "category": "shape", "gt_label": shape, "image": img,
                "filename": f"shape/{shape}_{idx:03d}.png",
                "pair_id": None, "pair_side": None,
            })
    return items


def gen_count_set():
    items = []
    r = 25; margin = r + 10; min_d = 60
    for n_str in PROMPTS["count"]["targets"]:
        n = int(n_str)
        for idx in range(50):
            random.seed(f"42-count-{n}-{idx}")
            placed = []
            cur_min_d = min_d
            for _ in range(n):
                ok = False
                for _ in range(100):
                    x = random.randint(margin, IMAGE_SIZE - margin)
                    y = random.randint(margin, IMAGE_SIZE - margin)
                    if all((x - px) ** 2 + (y - py) ** 2 >= cur_min_d ** 2 for px, py in placed):
                        placed.append((x, y)); ok = True; break
                if not ok:
                    cur_min_d = max(2 * r + 4, cur_min_d - 10)
                    for _ in range(100):
                        x = random.randint(margin, IMAGE_SIZE - margin)
                        y = random.randint(margin, IMAGE_SIZE - margin)
                        if all((x - px) ** 2 + (y - py) ** 2 >= cur_min_d ** 2 for px, py in placed):
                            placed.append((x, y)); ok = True; break
                if not ok:
                    placed.append((random.randint(margin, IMAGE_SIZE - margin),
                                   random.randint(margin, IMAGE_SIZE - margin)))
            img = _new_canvas(); d = ImageDraw.Draw(img)
            for x, y in placed:
                _draw_circle(d, x, y, r, fill=COLOR_RGB["red"])
            items.append({
                "category": "count", "gt_label": str(n), "image": img,
                "filename": f"count/{n}_{idx:03d}.png",
                "pair_id": None, "pair_side": None,
            })
    return items


def gen_spatial_set():
    # axis-aligned: only the relation axis varies between the two objects, so
    # ``above'' isn't contaminated by a horizontal offset.
    items = []
    Y = IMAGE_SIZE // 2; X = IMAGE_SIZE // 2
    base = {
        "left":  {"circle": (150, Y), "square": (298, Y), "axis": "x"},
        "right": {"circle": (298, Y), "square": (150, Y), "axis": "x"},
        "above": {"circle": (X, 150), "square": (X, 298), "axis": "y"},
        "below": {"circle": (X, 298), "square": (X, 150), "axis": "y"},
    }
    for rel in PROMPTS["spatial"]["targets"]:
        cb = base[rel]["circle"]; sb = base[rel]["square"]; axis = base[rel]["axis"]
        for idx in range(50):
            random.seed(f"42-spatial-{rel}-{idx}")
            jc = random.randint(-15, 15)
            js = random.randint(-15, 15)
            if axis == "x":
                cx, cy = cb[0] + jc, cb[1]
                sx, sy = sb[0] + js, sb[1]
            else:
                cx, cy = cb[0], cb[1] + jc
                sx, sy = sb[0], sb[1] + js
            # base + ±15 can't flip the relation, but assert anyway
            if rel == "left":  assert cx < sx
            elif rel == "right": assert cx > sx
            elif rel == "above": assert cy < sy
            elif rel == "below": assert cy > sy
            img = _new_canvas(); d = ImageDraw.Draw(img)
            _draw_square(d, sx, sy, 60, fill=COLOR_RGB["blue"])
            _draw_circle(d, cx, cy, 30, fill=COLOR_RGB["red"])
            items.append({
                "category": "spatial", "gt_label": rel, "image": img,
                "filename": f"spatial/{rel}_{idx:03d}.png",
                "pair_id": None, "pair_side": None,
            })
    return items


def gen_binding_set():
    # 200 pairs. A: red-circle / blue-square. B: blue-circle / red-square.
    items = []
    base_left = (150, 224); base_right = (298, 224)
    for idx in range(200):
        random.seed(f"42-binding-{idx}")
        jlx, jly = random.randint(-20, 20), random.randint(-20, 20)
        jrx, jry = random.randint(-20, 20), random.randint(-20, 20)
        lx, ly = base_left[0] + jlx, base_left[1] + jly
        rx, ry = base_right[0] + jrx, base_right[1] + jry
        img_a = _new_canvas(); da = ImageDraw.Draw(img_a)
        _draw_square(da, rx, ry, 60, fill=COLOR_RGB["blue"])
        _draw_circle(da, lx, ly, 30, fill=COLOR_RGB["red"])
        items.append({
            "category": "binding", "gt_label": "red", "image": img_a,
            "filename": f"binding/pair{idx:03d}_A.png",
            "pair_id": idx, "pair_side": "A",
        })
        img_b = _new_canvas(); db = ImageDraw.Draw(img_b)
        _draw_square(db, rx, ry, 60, fill=COLOR_RGB["red"])
        _draw_circle(db, lx, ly, 30, fill=COLOR_RGB["blue"])
        items.append({
            "category": "binding", "gt_label": "blue", "image": img_b,
            "filename": f"binding/pair{idx:03d}_B.png",
            "pair_id": idx, "pair_side": "B",
        })
    return items


def generate_all_images():
    print("[gen ] generating dataset...")
    t0 = time.time()
    items = (gen_color_set() + gen_shape_set() + gen_count_set()
             + gen_spatial_set() + gen_binding_set())
    print(f"[gen ] {len(items)} images in {time.time()-t0:.1f}s")
    counts = {}
    for it in items:
        counts[it["category"]] = counts.get(it["category"], 0) + 1
    for k, v in counts.items():
        print(f"        {k:8s}: {v}")
    return items


def load_model():
    print("[init] loading model + processor...")
    t0 = time.time()
    if DEVICE == "cuda":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=DTYPE, device_map="auto"
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=DTYPE
        ).to(DEVICE)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print(f"[init] loaded in {time.time()-t0:.1f}s")
    return model, processor


def build_inputs(model, processor, image, category):
    prompt_info = PROMPTS[category]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_info["question"]},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = text + prompt_info["prefill"]
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    return {k: v.to(model.device) for k, v in inputs.items()}


def get_target_word(category, gt_label):
    if category == "count":
        return COUNT_LABEL_TO_WORD[gt_label]
    return gt_label


def get_distractor_ids(category, target_word):
    others = CATEGORY_CLASSES[category]
    return {w: TARGET_TOKEN_IDS[w] for w in others if w != target_word}


def compute_layer_metrics(probs, target_word, category, tokenizer):
    # Reductions in fp32 — entropy underflows badly in fp16.
    probs_fp32 = probs.float()
    target_ids = list(TARGET_TOKEN_IDS[target_word])
    p_target = float(probs_fp32[target_ids].sum().item())
    distractors = get_distractor_ids(category, target_word)
    p_per_distractor = {
        w: float(probs_fp32[list(ids)].sum().item())
        for w, ids in distractors.items()
    }
    p_distractor = float(sum(p_per_distractor.values()))
    margin = p_target - max(p_per_distractor.values()) if p_per_distractor else p_target
    # rank = # vocab tokens whose prob > best target-token prob (rank 0 = argmax)
    target_max_prob = float(probs_fp32[target_ids].max().item())
    rank_target = int((probs_fp32 > target_max_prob).sum().item())
    log_p = torch.log(probs_fp32.clamp_min(1e-12))
    entropy = float(-(probs_fp32 * log_p).sum().item())
    top5_p, top5_id = torch.topk(probs_fp32, 5)
    top5_ids = top5_id.tolist()
    top5_probs = [float(x) for x in top5_p.tolist()]
    top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids]
    return {
        "p_target": p_target,
        "p_distractor": p_distractor,
        "margin": margin,
        "rank_target": rank_target,
        "entropy": entropy,
        "top5_tokens": top5_tokens,
        "top5_probs": top5_probs,
        "top5_ids": top5_ids,
        "p_per_distractor": p_per_distractor,
    }


def _get_num_hidden_layers(config):
    # transformers 4.53 keeps it on the root config; 4.55+ buries it in text_config.
    for cfg in (config, getattr(config, "text_config", None),
                getattr(config, "language_config", None)):
        if cfg is None:
            continue
        n = getattr(cfg, "num_hidden_layers", None)
        if n is not None:
            return int(n)
    raise AttributeError("could not find num_hidden_layers on config or its sub-configs")


def _resolve_lm_paths(model):
    # Qwen2-VL nests the language model differently across transformers releases:
    #   4.53.x  -> model.model.language_model.{layers,norm}, model.lm_head
    #   4.55.x+ -> sometimes model.language_model.model.{layers,norm}
    # Walk a few known candidates and fall back to scanning named_modules.
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None and hasattr(model, "language_model"):
        lm_head = getattr(model.language_model, "lm_head", None)
    if lm_head is None:
        raise AttributeError("could not locate lm_head on the model")

    candidates = []
    if hasattr(model, "model"):
        if hasattr(model.model, "language_model"):
            candidates.append(("model.model.language_model", model.model.language_model))
        if hasattr(model.model, "layers") and hasattr(model.model, "norm"):
            candidates.append(("model.model", model.model))
    if hasattr(model, "language_model"):
        lang = model.language_model
        if hasattr(lang, "model"):
            candidates.append(("model.language_model.model", lang.model))
        if hasattr(lang, "layers") and hasattr(lang, "norm"):
            candidates.append(("model.language_model", lang))

    for path, mod in candidates:
        if hasattr(mod, "layers") and hasattr(mod, "norm"):
            return path, mod, mod.norm, lm_head

    for name, mod in model.named_modules():
        if hasattr(mod, "layers") and hasattr(mod, "norm"):
            try:
                n = len(mod.layers)
            except TypeError:
                continue
            if n >= 24:
                return name, mod, mod.norm, lm_head

    raise AttributeError("could not locate the language-model decoder (.layers + .norm)")


def main():
    t_global = time.time()
    model, processor = load_model()
    tokenizer = processor.tokenizer

    lang_path, lang_mod, final_norm, lm_head = _resolve_lm_paths(model)
    n_layers = _get_num_hidden_layers(model.config)
    print(f"[paths] language_model = {lang_path}  (n_layers={n_layers})")
    print(f"[paths] final_norm = {type(final_norm).__name__}  lm_head = {type(lm_head).__name__}")
    if n_layers != NUM_LAYERS:
        print(f"[warn] config reports {n_layers} layers; rebinding NUM_LAYERS (was {NUM_LAYERS})")
        globals()["NUM_LAYERS"] = n_layers

    dataset = generate_all_images()
    max_images = os.environ.get("MAX_IMAGES")
    if max_images is not None:
        dataset = dataset[: int(max_images)]
        print(f"[init] MAX_IMAGES={max_images} -> {len(dataset)} images")

    # Stream rows to CSV so a Colab disconnect doesn't lose work.
    out_path = Path("logit_lens_results.csv")
    fields = [
        "category", "image_file", "gt_label", "target_word",
        "pair_id", "pair_side", "layer",
        "p_target", "p_distractor", "margin", "rank_target", "entropy",
        "top1_token", "top1_prob", "top1_token_id",
        "top2_token", "top2_prob", "top2_token_id",
        "top3_token", "top3_prob", "top3_token_id",
        "top4_token", "top4_prob", "top4_token_id",
        "top5_token", "top5_prob", "top5_token_id",
    ]
    f = open(out_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    print(f"\n[lens] sweeping {len(dataset)} images × {NUM_LAYERS+1} layers...")
    t_sweep = time.time()
    for i, item in enumerate(tqdm(dataset, desc="lens")):
        try:
            inputs = build_inputs(model, processor, item["image"], item["category"])
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            assert len(hidden_states) == NUM_LAYERS + 1
            target_word = get_target_word(item["category"], item["gt_label"])

            for layer_idx in range(NUM_LAYERS + 1):
                h = hidden_states[layer_idx][:, -1, :]
                # hidden_states[-1] is already post-norm in HF Qwen2-VL; the
                # earlier 28 entries are not.
                if layer_idx < NUM_LAYERS:
                    logits = lm_head(final_norm(h))
                else:
                    logits = lm_head(h)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                m = compute_layer_metrics(probs, target_word, item["category"], tokenizer)
                row = {
                    "category":     item["category"],
                    "image_file":   item["filename"],
                    "gt_label":     item["gt_label"],
                    "target_word":  target_word,
                    "pair_id":      item.get("pair_id") if item.get("pair_id") is not None else "",
                    "pair_side":    item.get("pair_side") or "",
                    "layer":        layer_idx,
                    "p_target":     m["p_target"],
                    "p_distractor": m["p_distractor"],
                    "margin":       m["margin"],
                    "rank_target":  m["rank_target"],
                    "entropy":      m["entropy"],
                }
                for j in range(5):
                    row[f"top{j+1}_token"]    = m["top5_tokens"][j]
                    row[f"top{j+1}_prob"]     = m["top5_probs"][j]
                    row[f"top{j+1}_token_id"] = m["top5_ids"][j]
                writer.writerow(row)

            del outputs, hidden_states
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            elif DEVICE == "mps":
                torch.mps.empty_cache()

            if (i + 1) % 100 == 0:
                f.flush()
                print(f"  [{i+1}/{len(dataset)}] last={item['filename']}  "
                      f"L28 P(target)={m['p_target']:.4f}  rank={m['rank_target']}")
        except Exception as e:
            print(f"  [err] {item['filename']}: {e!r}")
            continue

    f.close()
    elapsed = time.time() - t_global
    sweep_t = time.time() - t_sweep
    print(f"\n[done] sweep={sweep_t/60:.1f}min total={elapsed/60:.1f}min  csv={out_path}")

    df = pd.read_csv(out_path)
    print(f"\n[csv ] shape={df.shape}  expected={len(dataset)}×{NUM_LAYERS+1}={len(dataset)*(NUM_LAYERS+1)}")
    # pair_id / pair_side are empty for non-binding rows (pandas reads as NaN);
    # only check NaN/Inf on the real metric columns.
    metric_cols = [
        "p_target", "p_distractor", "margin", "rank_target", "entropy",
        "top1_prob", "top2_prob", "top3_prob", "top4_prob", "top5_prob",
        "top1_token_id", "top2_token_id", "top3_token_id", "top4_token_id", "top5_token_id",
    ]
    nan_count = df[metric_cols].isna().sum().sum()
    inf_count = int((df[metric_cols].abs() == math.inf).sum().sum())
    print(f"[csv ] NaN(metrics)={nan_count}  Inf(metrics)={inf_count}  "
          f"(empty pair_id/pair_side for non-binding rows are expected and not counted)")

    print("\n=== L28 accuracy (rank_target == 0) ===")
    for cat in ["color", "shape", "count", "spatial", "binding"]:
        sub = df[(df.category == cat) & (df.layer == NUM_LAYERS)]
        if len(sub) == 0:
            print(f"  {cat:>10}: (no rows)")
            continue
        acc = (sub.rank_target == 0).mean()
        print(f"  {cat:>10}: {acc:.1%}  (n={len(sub)})")

    print("\n=== Mean P(target) trajectory ===")
    layers_to_show = [0, 7, 14, 21, NUM_LAYERS]
    header = "category   |" + "".join(f"  L{L:02d}  " for L in layers_to_show)
    print(header)
    print("-" * len(header))
    for cat in ["color", "shape", "count", "spatial", "binding"]:
        sub = df[df.category == cat]
        if len(sub) == 0:
            continue
        row = f"{cat:>10} |"
        for L in layers_to_show:
            v = sub[sub.layer == L].p_target.mean()
            row += f"  {v:.3f}"
        print(row)

    try:
        from google.colab import files  # noqa: F401
        print(f"\n[colab] offering {out_path} for download")
        files.download(str(out_path))
    except Exception:
        print(f"\n[local] CSV saved at {out_path.resolve()}")


if __name__ == "__main__":
    main()
