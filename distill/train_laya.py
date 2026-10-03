#!/usr/bin/env python3
"""Fine-tune Laya for systematic-review abstract screening (PICO: T2DM exercise RCTs).

Trains the exact module layout that pagoda-hf's Rust loader expects, then exports
a drop-in checkpoint directory loadable via `laya_server --model-dir`.

Per seed abstract we build two typed questions, mirroring pagoda-hf/src/laya.rs:
  * noul  : "does the study meet the PICO inclusion criteria?" -> include/exclude
  * choice: study design in {rct, cohort, review, invitro, other}

Usage:
  python train_laya.py                      # train + recalibrate + export
  python train_laya.py --eval-only          # score the BASE checkpoint on the seeds
  python train_laya.py --eval-only --ckpt out/laya-screening   # score an export
"""
import argparse, glob, json, math, os, random, shutil, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

DESIGNS = [
    ("rct", "randomized controlled trial with random allocation of participants"),
    ("cohort", "observational cohort or case-control study without randomization"),
    ("review", "narrative or systematic review, meta-analysis, or pooled analysis"),
    ("invitro", "in-vitro, animal, or mechanistic laboratory study"),
    ("other", "other design: cross-sectional, case report, quasi-randomized, conference abstract"),
]
NOUL_OPTS = ["false: no, the statement does not hold", "true: yes, the statement holds"]
QTYPE_INDEX = {"choice": 0, "score": 1, "noul": 2}

NOUL_INSTR = ("Answer whether the statement about the scientific abstract is true. "
              "Statement: the study is a randomized controlled trial in adults with type 2 "
              "diabetes comparing a structured exercise intervention (aerobic, resistance, "
              "or combined training) against usual care or no exercise, and reports HbA1c "
              "(glycated hemoglobin) as an outcome.")
CHOICE_INSTR = "Classify the study design of the scientific abstract described in the text."


def build_sequence(tok, state, qtype, instructions, options, max_len, head_max_len,
                   cls, sep, mask):
    """Faithful Python port of pagoda-hf/src/laya.rs build_sequence."""
    def encode(text):
        return tok.encode(text.replace("[MASK]", " "), add_special_tokens=False)

    head_ids = encode(f"{qtype} question: {instructions}")
    opt_ids = []
    for o in options:
        rest = encode(" " + o)[:48]
        opt_ids.append([mask] + rest)
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max((head_max_len - 16) // max(len(opt_ids), 1), 4)
        opt_ids = [o[:per] for o in opt_ids]
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: min(max(opt_budget, 8), len(head_ids))]

    ids = [cls] + head_ids + [sep]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(sep)
    room = max(max_len - (len(ids) + 1), 0)
    ids.extend(encode(state)[:room])
    ids.append(sep)
    ids = ids[:max_len]
    markers = [m for m in markers if m < max_len]
    assert len(markers) == len(options), "options truncated out of the window"
    return ids, markers


class LayaScreening(nn.Module):
    """Module tree whose state_dict keys are exactly the checkpoint layout."""

    def __init__(self, encoder, d=1024, nhead=16):
        super().__init__()
        self.encoder = encoder
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=4 * d, activation="relu",
            batch_first=True, norm_first=True, dropout=0.0)
        self.head = nn.TransformerEncoder(layer, num_layers=2)
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d, eps=1e-5), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, input_ids, attn, qtypes, markers_list):
        h = self.encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
        h = h + self.type_emb(qtypes).unsqueeze(1)
        # Float key-padding mask: additive -1e9 on padded columns, mirroring the
        # Rust head (att.affine(-1,1).affine(-1e9,0)) via src_key_padding_mask.
        key_pad = (1.0 - attn.float()) * -1e9  # [b, l]
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=key_pad)
        outs = []
        for r, markers in enumerate(markers_list):
            m = h[r, markers]                      # [k, d]
            logits = self.scorer(m).squeeze(-1)    # [k]
            outs.append(logits)
        return outs


def load_checkpoint(ckpt_dir):
    from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel
    ckpt_dir = Path(ckpt_dir)
    tok = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer")
    cfg = ModernBertConfig.from_pretrained(ckpt_dir / "encoder")
    # Force SDPA and disable transformers' built-in torch.compile paths
    # (reference_compile wraps attention/MLP in inductor, which first-run
    # compiles for many minutes and can hang inside nohup pipelines).
    cfg._attn_implementation = "sdpa"
    cfg.reference_compile = False
    enc = ModernBertModel(cfg)
    model = LayaScreening(enc, d=cfg.hidden_size, nhead=max(cfg.hidden_size // 64, 1))
    sd = load_file(str(ckpt_dir / "model.safetensors"))
    sd.pop("temperature", None)
    missing, unexpected = model.load_state_dict(
        {k: v.float() for k, v in sd.items()}, strict=False)
    real_missing = [k for k in missing if not k.startswith("encoder._")]
    print(f"[load] missing={len(real_missing)} unexpected={len(unexpected)}")
    if real_missing:
        print("[load] missing sample:", real_missing[:6])
    if unexpected:
        print("[load] unexpected sample:", unexpected[:6])
    assert not unexpected, "checkpoint keys do not match the module tree"
    assert not real_missing, f"missing weights: {real_missing[:4]}"
    rl = json.loads((ckpt_dir / "rl_agent_config.json").read_text())
    return model, tok, rl


def make_examples(tok, seeds, max_len, head_max_len):
    cls = tok.convert_tokens_to_ids("[CLS]")
    sep = tok.convert_tokens_to_ids("[SEP]")
    mask = tok.convert_tokens_to_ids("[MASK]")
    examples = []
    for s in seeds:
        state = f"Title: {s['title']}\nAbstract: {s['abstract']}"
        ids, markers = build_sequence(tok, state, "noul", NOUL_INSTR, NOUL_OPTS,
                                      max_len, head_max_len, cls, sep, mask)
        examples.append(dict(ids=ids, markers=markers, qtype=QTYPE_INDEX["noul"],
                             qtype_name="noul", label=1 if s["include"] else 0,
                             seed=s["id"], borderline=bool(s.get("borderline"))))
        options = [f"{k}: {d}" for k, d in DESIGNS]
        didx = [k for k, _ in DESIGNS].index(s["design"])
        ids, markers = build_sequence(tok, state, "choice", CHOICE_INSTR, options,
                                      max_len, head_max_len, cls, sep, mask)
        examples.append(dict(ids=ids, markers=markers, qtype=QTYPE_INDEX["choice"],
                             qtype_name="choice", label=didx,
                             seed=s["id"], borderline=bool(s.get("borderline"))))
    return examples


def collate(batch, pad_id, device):
    lmax = max(len(e["ids"]) for e in batch)
    b = len(batch)
    ids = torch.full((b, lmax), pad_id, dtype=torch.long)
    att = torch.zeros((b, lmax), dtype=torch.float)
    for r, e in enumerate(batch):
        ids[r, : len(e["ids"])] = torch.tensor(e["ids"])
        att[r, : len(e["ids"])] = 1.0
    qtypes = torch.tensor([e["qtype"] for e in batch])
    markers = [e["markers"] for e in batch]
    labels = torch.tensor([e["label"] for e in batch])
    return (ids.to(device), att.to(device), qtypes.to(device), markers,
            labels.to(device))


def bucket(k):
    return "2" if k <= 2 else ("3-5" if k <= 5 else ("6-10" if k <= 10 else "11+"))


@torch.no_grad()
def evaluate(model, examples, pad_id, device, temps=None, bs=8):
    model.eval()
    rows = []
    for i in range(0, len(examples), bs):
        chunk = examples[i : i + bs]
        ids, att, qtypes, markers, labels = collate(chunk, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outs = model(ids, att, qtypes, markers)
        for e, logits, lab in zip(chunk, outs, labels):
            logits = logits.float().cpu()
            k = logits.numel()
            t = 1.0
            if temps is not None:
                t = temps.get(f"{e['qtype_name']}:{bucket(k)}", 1.0)
            p = torch.softmax(logits / t, dim=0)
            rows.append(dict(qtype=e["qtype_name"], label=int(lab), k=k,
                             pred=int(p.argmax()), conf=float(p.max()),
                             ptrue=float(p[e['label']]),
                             borderline=e["borderline"], seed=e["seed"]))
    return rows


def accuracy(rows, qtype=None, borderline=None):
    sel = [r for r in rows
           if (qtype is None or r["qtype"] == qtype)
           and (borderline is None or r["borderline"] == borderline)]
    if not sel:
        return float("nan"), 0
    return sum(r["pred"] == r["label"] for r in sel) / len(sel), len(sel)


def ece(rows, bins=5):
    sel = [r for r in rows if r["qtype"] == "noul"]
    if not sel:
        return float("nan")
    total, acc_gap = len(sel), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        grp = [r for r in sel if lo <= r["conf"] < hi or (b == bins - 1 and r["conf"] == 1.0)]
        if grp:
            acc = sum(r["pred"] == r["label"] for r in grp) / len(grp)
            conf = sum(r["conf"] for r in grp) / len(grp)
            acc_gap += len(grp) * abs(acc - conf)
    return acc_gap / total


def report(tag, rows):
    a_n, n_n = accuracy(rows, "noul")
    a_nb, n_nb = accuracy(rows, "noul", borderline=True)
    a_c, n_c = accuracy(rows, "choice")
    print(f"[{tag}] include(noul) acc={a_n:.3f} (n={n_n}) | "
          f"borderline acc={a_nb:.3f} (n={n_nb}) | design(choice) acc={a_c:.3f} (n={n_c}) | "
          f"ECE(noul)={ece(rows):.3f}")
    return dict(noul=a_n, noul_borderline=a_nb, choice=a_c, ece=ece(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint dir (default: HF cache snapshot)")
    ap.add_argument("--seeds", default=str(Path(__file__).parent / "seeds" / "abstracts.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "out" / "laya-screening"))
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)

    ckpt = args.ckpt
    if ckpt is None:
        hits = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/*"))
        assert hits, "base checkpoint not in HF cache; pass --ckpt"
        ckpt = hits[0]
    print(f"[setup] checkpoint: {ckpt}")

    model, tok, rl = load_checkpoint(ckpt)
    max_len, head_max_len = rl["max_len"], rl["head_max_len"]
    pad_id = tok.convert_tokens_to_ids("[PAD]")

    seeds = [json.loads(l) for l in Path(args.seeds).read_text().splitlines() if l.strip()]
    pico = next(s["pico"] for s in seeds if "pico" in s)
    seeds = [s for s in seeds if "id" in s]
    print(f"[setup] {pico}")
    print(f"[setup] {len(seeds)} seed abstracts")

    examples = make_examples(tok, seeds, max_len, head_max_len)
    # Deterministic stratified split: every 5th seed per (design,include) group -> val.
    groups = {}
    for s in seeds:
        groups.setdefault((s["design"], s["include"]), []).append(s["id"])
    val_ids = {sid for ids in groups.values() for i, sid in enumerate(sorted(ids)) if i % 5 == 4}
    train_ex = [e for e in examples if e["seed"] not in val_ids]
    val_ex = [e for e in examples if e["seed"] in val_ids]
    print(f"[setup] train={len(train_ex)} val={len(val_ex)} questions "
          f"({len(val_ids)} val abstracts)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.eval_only:
        temps = rl.get("temperature_by_options", {})
        rows = evaluate(model, val_ex or examples, pad_id, device, temps=temps)
        report("eval/" + ("val" if val_ex else "all"), rows)
        return

    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      target_modules=["Wqkv", "Wo", "Wi"], bias="none")
    model.encoder = get_peft_model(model.encoder, lcfg)
    # Gradient checkpointing keeps the 3050's spare VRAM comfortable even when
    # other tenants share the card. use_reentrant=False is required with PEFT.
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.encoder.enable_input_require_grads()
    n_lora = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    for name in ["head", "type_emb", "scorer"]:
        for p in getattr(model, name).parameters():
            p.requires_grad_(True)
    for p in model.act_head.parameters():
        p.requires_grad_(False)
    n_head = sum(p.numel() for n in ["head", "type_emb", "scorer"]
                 for p in getattr(model, n).parameters())
    print(f"[train] LoRA params={n_lora/1e6:.1f}M head params={n_head/1e6:.1f}M "
          f"(act_head frozen)")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_ex) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        random.shuffle(train_ex)
        model.train()
        tot = 0.0
        for i in range(0, len(train_ex), args.batch):
            chunk = train_ex[i : i + args.batch]
            ids, att, qtypes, markers, labels = collate(chunk, pad_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outs = model(ids, att, qtypes, markers)
            loss = sum(F.cross_entropy(o.float().unsqueeze(0), l.unsqueeze(0))
                       for o, l in zip(outs, labels)) / len(chunk)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            tot += float(loss)
        print(f"[train] epoch {ep+1}/{args.epochs} loss={tot/steps_per_epoch:.4f} "
              f"({time.time()-t0:.0f}s)")

    rows_val = evaluate(model, val_ex, pad_id, device)
    m = report("tuned/val(pre-temp)", rows_val)

    # --- temperature recalibration on val (NLL grid search per qtype:bucket) ---
    temps = dict(rl.get("temperature_by_options", {}))
    tuned = {}
    for e in val_ex:
        tuned.setdefault(f"{e['qtype_name']}:{bucket(len(e['markers']))}", []).append(e)
    for key, exs in tuned.items():
        rows = evaluate(model, exs, pad_id, device)
        best_t, best_nll = 1.0, float("inf")
        for t in [0.3 + 0.1 * i for i in range(38)]:
            nll = -sum(math.log(max(r["ptrue"], 1e-9)) for r in
                       evaluate(model, exs, pad_id, device, temps={key: t})) / len(rows)
            if nll < best_nll:
                best_nll, best_t = nll, t
        temps[key] = best_t
        print(f"[calib] {key}: T={best_t:.2f} (n={len(exs)}, nll={best_nll:.3f})")
    rows_cal = evaluate(model, val_ex, pad_id, device, temps=temps)
    m = report("tuned/val(calibrated)", rows_cal)

    # --- export: merge LoRA, save in the exact checkpoint layout ---
    model.encoder = model.encoder.merge_and_unload()
    model.to("cpu")
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "tokenizer").mkdir(parents=True)
    (out / "encoder").mkdir()
    sd = {k: v.contiguous().half() for k, v in model.state_dict().items()}
    base_temp = load_file(str(Path(ckpt) / "model.safetensors")).get("temperature")
    if base_temp is not None:
        sd["temperature"] = base_temp
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    shutil.copytree(Path(ckpt) / "tokenizer", out / "tokenizer", dirs_exist_ok=True)
    shutil.copy(Path(ckpt) / "encoder" / "config.json", out / "encoder" / "config.json")
    rl["temperature_by_options"] = temps
    rl["training"] = dict(updates=step, epochs_completed=args.epochs,
                          hours=round((time.time() - t0) / 3600, 3),
                          task="t2dm-exercise-abstract-screening",
                          seeds=len(seeds), fine_tuned_from_checkpoint=True,
                          val_metrics=m)
    (out / "rl_agent_config.json").write_text(json.dumps(rl, indent=2))
    print(f"[export] wrote {out} ({(out/'model.safetensors').stat().st_size/1e6:.0f} MB)")
    print("[export] serve with: laya_server --model-dir", out)


if __name__ == "__main__":
    main()
