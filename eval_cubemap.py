import os, glob, math, json
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import py360convert

#reaproveitar o que já existe em eval.py
from eval import (
    load_checkpoint, pad_centering, unpad_centering,
    compute_psnr, compute_msssim_db, read_image
)
from models import LALIC

def compute_ws_psnr(a, b):
    H = a.size(2)
    lat = torch.linspace(-math.pi/2, math.pi/2, H, device=a.device)
    weights = torch.cos(lat).view(1, 1, H, 1)
    weights = weights / weights.mean()
    mse = (weights * (a - b) ** 2).mean().item()
    return -10 * math.log10(mse)

FACE_SIZE = 512  #tamanho de cada uma das 6 faces do cubemap

@torch.no_grad()
def compress_decompress_erp_via_cubemap(model, erp_path, device):
    #carrega erp e projeta em 6 faces
    erp = np.array(Image.open(erp_path).convert("RGB")) / 255.0
    faces = py360convert.e2c(erp, face_w=FACE_SIZE, mode='bilinear', cube_format='dict')
    #faces é um dict com chaves 'F','R','B','L','U','D'

    total_bits = 0
    recon_faces = {}
    for key, face in faces.items():
        x = torch.from_numpy(face).permute(2, 0, 1).float().unsqueeze(0).to(device)
        x_padded, padding = pad_centering(x, 128)

        out_enc = model.compress(x_padded)
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        out_dec["x_hat"].clamp_(0, 1)
        x_hat = unpad_centering(out_dec["x_hat"], padding)

        bits = sum(len(s[0]) * 8.0 for s in out_enc["strings"])
        total_bits += bits

        recon_faces[key] = x_hat.squeeze(0).permute(1, 2, 0).cpu().numpy()

    #reconstrói a erp a partir das faces recuperadas
    h, w, _ = erp.shape
    recon_erp = py360convert.c2e(recon_faces, h=h, w=w, mode='bilinear', cube_format='dict')
    recon_erp = np.clip(recon_erp, 0, 1)

    x_orig = torch.from_numpy(erp).permute(2, 0, 1).float().unsqueeze(0).to(device)
    x_recon = torch.from_numpy(recon_erp).permute(2, 0, 1).float().unsqueeze(0).to(device)

    num_pixels = h * w
    bpp = total_bits / num_pixels
    psnr = compute_psnr(x_recon, x_orig)
    ws_psnr = compute_ws_psnr(x_recon, x_orig)
    ms_ssim_db = compute_msssim_db(x_recon, x_orig)

    return {"bpp": bpp, "psnr": psnr, "ws-psnr": ws_psnr, "ms-ssim-db": ms_ssim_db}


device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_checkpoint(LALIC, "checkpoints_raw/lalic-q3.pth").to(device)
model.update()  #antes de usar compress/decompress real

results = []
for f in sorted(glob.glob("dataset360_teste5/*.jpg")):
    r = compress_decompress_erp_via_cubemap(model, f, device)
    print(f, r)
    results.append(r)

avg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
print("MÉDIA:", avg)