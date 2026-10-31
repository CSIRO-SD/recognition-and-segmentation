#!/usr/bin/env python3
"""Zero-shot on your CSV using **official** [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP) (``eva_clip``), not HuggingFace.

Mirrors the upstream README::

    model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
    tokenizer = get_tokenizer(model_name)
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    text = tokenizer(captions).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

This repo does **not** vendor ``eva_clip``. Install / clone BAAI EVA-CLIP and put it on ``PYTHONPATH``, e.g.::

    git clone https://github.com/baaivision/EVA.git
    export PYTHONPATH=\"/path/to/EVA/EVA-CLIP:${PYTHONPATH}\"

(or ``pip install -e .`` from the EVA-CLIP folder if supported).

**HF vs official:** ``BAAI/EVA-CLIP-8B`` on HuggingFace is a different port (Transformers + remote code).
For paper-faithful EVA-CLIP, use the **official** ``model_name`` / ``pretrained`` from the EVA-CLIP README
(``EVA02-CLIP-B-16`` + ``pretrained=eva_clip``, or a local ``.pt`` path). There is no 1:1 name for every Hub id.

If ``pip install xformers`` gave a **CPU-only** wheel (CUDA extensions missing), EVA's ``xattn`` path raises
``NotImplementedError: No operator found for memory_efficient_attention_forward``. This script defaults to
``--xattn auto``: it probes xFormers on GPU and, if broken, patches attention modules to use PyTorch.
Use ``--xattn off`` to force that path, or reinstall xFormers built for your PyTorch/CUDA version.

Example::

  PYTHONPATH=/path/to/EVA/EVA-CLIP:$PYTHONPATH \\
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_eva_clip_official.py \\
    --csv ... --images-root ... \\
    --eval-split test_seen \\
    --prompt-col label_text \\
    --model-name EVA02-CLIP-B-16 \\
    --pretrained eva_clip \\
    --hierarchy-report-prefix ./dbg_eva_official/B16/zeroshot

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clip_siglip.data.taxonomy_csv import TaxonomyTable
from clip_siglip.eval.zeroshot import (
    build_class_prompts,
    build_hierarchy_by_class_col,
    zeroshot_metrics_pil,
)
from clip_siglip.eval.zeroshot import _norm_cell
from clip_siglip.models.wrappers import VLBackbone


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _probe_xformers_cuda_attention() -> bool:
    """True if ``xformers.ops.memory_efficient_attention`` runs on this GPU build (smoke test)."""
    if not torch.cuda.is_available():
        return False
    try:
        import xformers.ops as xops  # noqa:PLC0415
    except ImportError:
        return False
    try:
        q = torch.zeros(1, 2, 8, 64, device="cuda", dtype=torch.float16)
        xops.memory_efficient_attention(q, q, q)
    except Exception:
        return False
    return True


def _resolve_use_xattn(mode: Literal["auto", "on", "off"]) -> tuple[bool, str]:
    """Return (inner_attention_modules_use_xformers, log_reason)."""
    if mode == "on":
        return True, "xattn on (--xattn on)"
    if mode == "off":
        return False, "xattn off (--xattn off)"
    if os.environ.get("EVA_CLIP_XATTN", "").lower() in ("0", "false", "off", "no"):
        return False, "xattn off (EVA_CLIP_XATTN)"
    if os.environ.get("EVA_CLIP_XATTN", "").lower() in ("1", "true", "on", "yes"):
        return True, "xattn on (EVA_CLIP_XATTN)"
    if _probe_xformers_cuda_attention():
        return True, "xattn on (auto: xFormers CUDA OK)"
    return False, "xattn off (auto: xFormers CUDA missing or broken; PyTorch attention)"


def _apply_eva_xattn_choice(model: nn.Module, *, use_xformers: bool) -> int:
    """Toggle EVA-CLIP ``Attention`` / ``CustomAttention`` modules; returns number patched."""
    if use_xformers:
        return 0
    n = 0
    for m in model.modules():
        if type(m).__name__ in ("Attention", "CustomAttention") and hasattr(m, "xattn"):
            m.xattn = False
            n += 1
    return n


class EVAOfficialVL(VLBackbone):
    """``eva_clip`` model + official ``preprocess`` / ``get_tokenizer`` (L2-normalized outputs)."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        *,
        device: torch.device | None = None,
        force_custom_clip: bool = True,
        autocast_cuda: bool = True,
        xattn: Literal["auto", "on", "off"] = "auto",
    ):
        try:
            from eva_clip import create_model_and_transforms, get_tokenizer
        except ImportError as e:
            raise ImportError(
                "Cannot import ``eva_clip``. Clone BAAI EVA and add EVA-CLIP to PYTHONPATH, e.g.\n"
                "  export PYTHONPATH=\"/path/to/EVA/EVA-CLIP:${PYTHONPATH}\"\n"
                "See https://github.com/baaivision/EVA/tree/master/EVA-CLIP",
            ) from e

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_autocast = bool(autocast_cuda and self.device.type == "cuda")

        model, _, preprocess = create_model_and_transforms(
            model_name,
            pretrained,
            force_custom_clip=force_custom_clip,
        )
        self.model = model.to(self.device).eval()
        use_xf, xattn_reason = _resolve_use_xattn(xattn)
        self._xattn_mode_log = xattn_reason
        patched = _apply_eva_xattn_choice(self.model, use_xformers=use_xf)
        self._xattn_patched_modules = patched
        self.preprocess = preprocess
        self.tokenizer = get_tokenizer(model_name)
        self.processor = None  # not used; satisfies optional introspection

    def _maybe_autocast(self):
        if self._use_autocast:
            return torch.cuda.amp.autocast()
        return torch.cuda.amp.autocast(enabled=False)

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device)
        with self._maybe_autocast():
            feat = self.model.encode_image(pixel_values)
        return self.l2norm(feat.float())

    @torch.no_grad()
    def encode_images_pil(self, images: list) -> torch.Tensor:
        ims = [im.convert("RGB") if hasattr(im, "convert") else im for im in images]
        batch = torch.stack([self.preprocess(im) for im in ims]).to(self.device)
        with self._maybe_autocast():
            feat = self.model.encode_image(batch)
        return self.l2norm(feat.float())

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        # Official: tokenizer(list[str]) -> LongTensor [B, context_length]
        tok = self.tokenizer(texts)
        if not isinstance(tok, torch.Tensor):
            tok = torch.as_tensor(tok)
        tok = tok.to(self.device)
        with self._maybe_autocast():
            feat = self.model.encode_text(tok)
        return self.l2norm(feat.float())


def _print_load_summary(backbone: EVAOfficialVL, model_name: str, pretrained: str) -> None:
    print("[eva_clip official] Loaded", file=sys.stderr)
    print(f"  --model-name:  {model_name!r}", file=sys.stderr)
    print(f"  --pretrained:  {pretrained!r}", file=sys.stderr)
    print(f"  torch module:  {type(backbone.model).__name__}", file=sys.stderr)
    print(f"  device:        {backbone.device}", file=sys.stderr)
    print(f"  attention:     {backbone._xattn_mode_log}", file=sys.stderr)
    if backbone._xattn_patched_modules:
        print(f"  patched attn:  {backbone._xattn_patched_modules} module(s) -> PyTorch path", file=sys.stderr)


def _readme_demo(backbone: EVAOfficialVL, image_path: Path, device: torch.device) -> None:
    image = Image.open(image_path).convert("RGB")
    captions = ["a diagram", "a dog", "a cat"]
    t = backbone.preprocess(image).unsqueeze(0).to(device)
    txt = backbone.tokenizer(captions).to(device)
    with torch.no_grad(), backbone._maybe_autocast():
        imf = backbone.model.encode_image(t)
        txf = backbone.model.encode_text(txt)
        imf = imf / imf.norm(dim=-1, keepdim=True)
        txf = txf / txf.norm(dim=-1, keepdim=True)
        probs = (100.0 * imf @ txf.T).softmax(dim=-1)
    print("[demo] README-style (100 * cosine @ softmax)")
    print(f"  image: {image_path}")
    print(f"  probs: {probs.float().cpu().tolist()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--images-root", required=True, type=Path)
    p.add_argument("--species-split", default="", help="Optional species_split filter (comma-separated).")
    p.add_argument("--eval-split", default="", help="e.g. test_seen,test_unseen")
    p.add_argument(
        "--model-name",
        default="EVA02-CLIP-B-16",
        help="Official EVA-CLIP id, e.g. EVA02-CLIP-B-16, EVA02-CLIP-L-14 (see EVA-CLIP README).",
    )
    p.add_argument(
        "--pretrained",
        default="eva_clip",
        help="``eva_clip`` to download per upstream, or path to a ``.pt`` checkpoint.",
    )
    p.add_argument("--no-force-custom-clip", action="store_true", help="Pass force_custom_clip=False.")
    p.add_argument("--no-autocast", action="store_true", help="Disable CUDA autocast (debug / fp32).")
    p.add_argument(
        "--xattn",
        choices=("auto", "on", "off"),
        default="auto",
        help="EVA memory_efficient_attention (xFormers): auto probes GPU; off uses PyTorch matmul (fixes "
        "NotImplementedError when xFormers has no CUDA). Env override: EVA_CLIP_XATTN=0|1 when mode is auto.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--scientific-name-col", default="scientific_name")
    p.add_argument("--taxon-col", default="taxon_key")
    p.add_argument("--prompt-col", default=None, metavar="COL")
    p.add_argument("--eval-class-col", default=None, metavar="COL")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--hierarchy-report-prefix", type=Path, default=None, metavar="PREFIX")
    p.add_argument("--hierarchy-debug", action="store_true")
    p.add_argument("--show-model", action="store_true", help="Print load summary to stderr.")
    p.add_argument("--demo-only", action="store_true")
    p.add_argument("--demo-image", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = EVAOfficialVL(
        args.model_name,
        args.pretrained,
        device=device,
        force_custom_clip=not args.no_force_custom_clip,
        autocast_cuda=not args.no_autocast,
        xattn=args.xattn,  # type: ignore[arg-type]
    )
    if backbone._xattn_patched_modules:
        print(
            f"[eva_clip official] {backbone._xattn_mode_log} (patched {backbone._xattn_patched_modules} attention modules).",
            file=sys.stderr,
        )
    if args.show_model:
        _print_load_summary(backbone, args.model_name, args.pretrained)

    if args.demo_only:
        img_p = args.demo_image
        if img_p is None:
            print("Use --demo-image PATH.png with --demo-only (or run full eval).", file=sys.stderr)
            sys.exit(1)
        _readme_demo(backbone, Path(img_p), device)
        return

    ss = _parse_csv_list(args.species_split) or None
    eval_parts = _parse_csv_list(args.eval_split)
    if not eval_parts:
        eval_parts = [None]
    if ss is None and eval_parts == [None]:
        print("Specify --species-split and/or --eval-split.", file=sys.stderr)
        sys.exit(1)

    table = TaxonomyTable.from_csv(args.csv, images_root=args.images_root)

    for ep in eval_parts:
        split_assign = [ep] if ep else None
        df = table.subset(species_splits=ss, split_assignments=split_assign)
        if len(df) == 0:
            print(f"No rows for eval_split={ep!r}", file=sys.stderr)
            sys.exit(1)

        if args.eval_class_col:
            class_col = args.eval_class_col
        elif args.prompt_col:
            class_col = args.prompt_col
        else:
            class_col = args.taxon_col
        if class_col not in df.columns:
            print(f"Missing column {class_col!r}", file=sys.stderr)
            sys.exit(1)

        sci_col = args.scientific_name_col if args.scientific_name_col in df.columns else None
        class_prompts = build_class_prompts(
            df,
            class_col=class_col,
            scientific_name_col=sci_col,
            prompt_col=args.prompt_col,
        )
        hierarchy = build_hierarchy_by_class_col(df, group_col=class_col)

        paths: list[Path] = []
        gt_samples: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            paths.append(Path(args.images_root) / str(row["image_name"]))
            gt_samples.append(
                {
                    "eval_class": _norm_cell(row[class_col]),
                    "order": row["order"] if "order" in df.columns else "",
                    "family": row["family"] if "family" in df.columns else "",
                    "genus": row["genus"] if "genus" in df.columns else "",
                    "species": row["species"] if "species" in df.columns else "",
                }
            )

        report_pfx = args.hierarchy_report_prefix
        if report_pfx is not None and ep:
            report_pfx = report_pfx.parent / f"{report_pfx.name}_{ep}"

        metrics = zeroshot_metrics_pil(
            backbone,
            class_prompts,
            paths,
            hierarchy,
            gt_samples,
            batch_size=args.batch_size,
            topk=args.topk,
            top1_class_label=class_col,
            hierarchy_report_prefix=report_pfx,
            hierarchy_debug_stderr=args.hierarchy_debug,
        )

        ss_part = ",".join(ss) if ss else "*"
        ep_part = ep or "*"
        print(
            f"EVA-CLIP official  model_name={args.model_name!r}  pretrained={args.pretrained!r}  "
            f"species_split=[{ss_part}]  csv_split=[{ep_part}]",
        )
        for line in metrics.lines():
            print(line)


if __name__ == "__main__":
    main()
