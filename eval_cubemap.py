"""
Modelo lalic com ERP -> cubemap -> ERP  (sem retreino).
Espelha a estrutura de eval.py, mas comprime cada imagem através das 6 faces do cubemap.
"""

import os
import re
import sys
import time
import glob
import math
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
import py360convert

import compressai
from compressai.registry import MODELS

from eval import (
    load_checkpoint, pad_centering, unpad_centering,
    compute_psnr, compute_msssim_db, reglob_collect_images,
)
from models import LALIC

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)


def compute_ws_psnr(a, b):
    H = a.size(2)
    lat = torch.linspace(-math.pi / 2, math.pi / 2, H, device=a.device)
    weights = torch.cos(lat).view(1, 1, H, 1)
    weights = weights / weights.mean()
    mse = (weights * (a - b) ** 2).mean().item()
    return -10 * math.log10(mse)


img_metrics = {
    "psnr": compute_psnr,
    "ms-ssim-db": compute_msssim_db,
    "ws-psnr": compute_ws_psnr,
}


@torch.no_grad()
def inference_cubemap(model, erp_path, face_size, fout=""):
    erp = np.array(Image.open(erp_path).convert("RGB")) / 255.0
    device = next(model.parameters()).device

    start = time.time()
    #projeta erp em 6 faces cubemap
    faces = py360convert.e2c(erp, face_w=face_size, mode='bilinear', cube_format='dict')
    # devolve {'F': array, 'R': array, 'B': array, 'L': array, 'U': array, 'D': array}
    total_bits = 0
    recon_faces = {}
    enc_time_total = 0.0
    dec_time_total = 0.0

    for key, face in faces.items():
        x = torch.from_numpy(face).permute(2, 0, 1).float().unsqueeze(0).to(device)
        x_padded, padding = pad_centering(x, 128)

        t0 = time.time()
        out_enc = model.compress(x_padded)
        enc_time_total += time.time() - t0

        t0 = time.time()
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        dec_time_total += time.time() - t0

        out_dec["x_hat"].clamp_(0, 1)
        x_hat = unpad_centering(out_dec["x_hat"], padding)

        #acumula os bits gastos (pra somar o custo total das 6 faces no final) e guarda a
        #  face reconstruída de volta em formato numpy
        bits = sum(len(s[0]) * 8.0 for s in out_enc["strings"])
        total_bits += bits
        recon_faces[key] = x_hat.squeeze(0).permute(1, 2, 0).cpu().numpy()

    h, w, _ = erp.shape
    #remontar erp a partir das 6 faces reconstruídas
    recon_erp = py360convert.c2e(recon_faces, h=h, w=w, mode='bilinear', cube_format='dict')
    recon_erp = np.clip(recon_erp, 0, 1)

    if fout:
        Image.fromarray((recon_erp * 255).astype(np.uint8)).save(fout)

    x_orig = torch.from_numpy(erp).permute(2, 0, 1).float().unsqueeze(0).to(device)
    x_recon = torch.from_numpy(recon_erp).permute(2, 0, 1).float().unsqueeze(0).to(device)

    num_pixels = h * w
    bpp = total_bits / num_pixels

    iqa_result = {key: func(x_recon, x_orig) for key, func in img_metrics.items()}
    org_result = {
        "enc_time": enc_time_total,
        "dec_time": dec_time_total,
        "bpp": bpp,
    }
    org_result.update(iqa_result)
    return org_result


#função que itera sobre todas as imagens do dataset, pra um checkpoint/qualidade específico
def eval_model(model, quality, args):
    avg_metrics = defaultdict(float)
    records = []

    filepaths = reglob_collect_images(args.input_dir)
    if len(filepaths) == 0:
        print(f"Error: no images found in {args.input_dir}.", file=sys.stderr)
        raise SystemExit(1)
    if args.output_dir:
        out_sub_dir = f"{args.output_dir}/{quality}"
        os.makedirs(out_sub_dir, exist_ok=True)

    model.update()  # obrigatório antes do compress/decompress real
    #Congela as tabelas de probabilidade do modelo
    for file in filepaths:
        fout = ""
        if args.output_dir:
            fout = os.path.join(out_sub_dir, os.path.basename(file))

        #chama a função de compressão/descompressão do cubemap para cada imagem, e acumula os resultados
        rv = inference_cubemap(model, file, args.face_size, fout)
        for k, v in rv.items():
            avg_metrics[k] += v

        record = {"file": file, "quality": quality}
        if args.verbose:
            _rv = {key: round(value, 4) for key, value in rv.items()}
            print(file, _rv)
        rv = {key: round(value, 8) for key, value in rv.items()}
        record.update(rv)
        records.append(record)

    for k, v in avg_metrics.items():
        avg_metrics[k] = round(v / len(filepaths), 6)
    return avg_metrics, records


def setup_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="model architecture",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="enable CUDA")
    parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parser.add_argument(
        "-p",
        "--checkpoint",
        dest="checkpoints",
        type=str,
        nargs="*",
        help="checkpoint path list",
    )
    parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        nargs="*",
        required=True,
        help="quality labels correspoding to the checkpoint path list",
    )
    parser.add_argument("-i", "--input-dir", type=str)
    parser.add_argument("-o", "--output-dir", type=str, default="")
    parser.add_argument("-r", "--result", type=str, help="result file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    parser.add_argument("--face-size", type=int, default=512, help="tamanho de cada face do cubemap")
    return parser

def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)
    assert len(args.checkpoints) == len(args.qualities), "checkpoint and quality labels mismatch"

    if args.model in MODELS:
        net_cls = MODELS[args.model]
    else:
        raise ValueError(f"Model {args.model} not found.")

    all_avg_metrics = defaultdict(list)
    all_records = []
    for ckpt, quality in zip(args.checkpoints, args.qualities):
        if args.verbose:
            print(f"\nEvaluating ckpt: {ckpt}")
        model = load_checkpoint(net_cls, ckpt)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")

        avg_metrics, records = eval_model(model, quality, args)
        all_records.extend(records)
        for k, v in avg_metrics.items():
            all_avg_metrics[k].append(v)

        mem = torch.cuda.max_memory_allocated(device=None)
        print(f"GPU mem: \t{mem / (2**30):.3f} GB")
        del model

    result = {
        "name": f"{args.model}-cubemap",
        "description": f"{args.model} via ERP->Cubemap->ERP, cuda: {args.cuda}, quality: {args.qualities}",
        "results": all_avg_metrics,
    }
    print(json.dumps(result, indent=2))
    result["records"] = all_records
    if args.result:
        with open(args.result, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])