# evaluate.py
import os
import random
import torch
import numpy as np
from skimage import io, color
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

from utils.perceptual import PerceptualHashValue


# ----------------------------
# Paths (edit he_dir if needed)
# ----------------------------
os.makedirs("Metrics/test", exist_ok=True)

####gradpspstain



targ_dir = '/home/sri/Desktop/GradPSPStain/results/GradPSPStain/val_latest/images/real_B'   # real IHC
pred_dir = '/home/sri/Desktop/GradPSPStain/results/GradPSPStain/val_latest/images/fake_B'   # generated IHC
he_dir   = '/home/sri/Desktop/GradPSPStain/results/GradPSPStain/val_latest/images/real_A'   # H&E (for Rc)


# ----------------------------
# Image list (shared stem names)
# ----------------------------
img_list = [f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
assert len(img_list) > 0, "No images found in pred_dir"
img_ext = '.' + img_list[0].split('.')[-1]
stems = [os.path.splitext(f)[0] for f in img_list]

# Keep only stems that exist in all three folders (pred, real_B, real_A)
def has_all(s):
    return (os.path.exists(os.path.join(pred_dir, s + img_ext))
            and os.path.exists(os.path.join(targ_dir, s + img_ext))
            and os.path.exists(os.path.join(he_dir,   s + img_ext)))
stems = [s for s in stems if has_all(s)]
random.seed(0)
random.shuffle(stems)
print(f"[INFO] Evaluating {len(stems)} image pairs.")

# ----------------------------
# Helper: grayscale + float in [0,1]
# ----------------------------
def to_gray01(img):
    if img.ndim == 3:
        # assume RGB
        g = color.rgb2gray(img)
    else:
        g = img.astype(np.float32)
    g = g.astype(np.float32)
    if g.max() > 1.0:
        g = g / 255.0
    return np.clip(g, 0.0, 1.0)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ----------------------------
# FID
# ----------------------------
num_avail_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
num_workers = min(num_avail_cpus or 8, 8)

real_paths = [os.path.join(targ_dir, s + img_ext) for s in stems]
fake_paths = [os.path.join(pred_dir, s + img_ext) for s in stems]

dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device)

m1, s1 = calculate_activation_statistics(real_paths, model, batch_size=10, dims=dims,
                                         device=device, num_workers=num_workers)
m2, s2 = calculate_activation_statistics(fake_paths, model, batch_size=10, dims=dims,
                                         device=device, num_workers=num_workers)
fid_value = calculate_frechet_distance(m1, s1, m2, s2)
print(f'FID: {fid_value:.2f}')

# =========================================================
# KID (no external script)
# =========================================================
from utils.kid_score import compute_kid
kid_mean, kid_std = compute_kid(targ_dir,pred_dir,batch_size=50,dims=2048,device='cuda')
print(f"KID_mean: {kid_mean:.6f} | KID_std: {kid_std:.6f}")

# print(f"KID_mean     : {kid_mean:.6f}")
# print(f"KID_std      : {kid_std:.6f}")
print(f"KID_mean x1000 : {kid_mean * 1000:.6f}")



from skimage.color import rgb2hed



# =========================================================
# Optical Density (OD)
# =========================================================
def convert_RGB_to_OD(I):
    I = I.astype(np.float32)
    I[I == 0] = 1
    return np.maximum(-np.log(I / 255.0), 1e-6)

def compute_IoD(rgb):
    OD = convert_RGB_to_OD(rgb)
    return np.sum(np.mean(OD, axis=2))


# def to_gray01(img):
#     if img.ndim == 3:
#         img = color.rgb2gray(img)
#     img = img.astype(np.float32)
#     if img.max() > 1:
#         img /= 255.0
#     return img

def pearson_corr(a, b):
    a = a.ravel()
    b = b.ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])



# ----------------------------
# PHV
# ----------------------------
layers = ['layer_1', 'layer_2', 'layer_3', 'layer_4']
PHV = PerceptualHashValue(
    T=0.01, network='resnet50', layers=layers,
    resize=False, resize_mode='bilinear', instance_normalized=False
).to(device)

all_phv = []
for s in tqdm(stems, desc="PHV"):
    fake = io.imread(os.path.join(pred_dir, s + img_ext))
    real = io.imread(os.path.join(targ_dir, s + img_ext))
    fake = to_tensor(fake).to(device)
    real = to_tensor(real).to(device)
    phv_list = PHV(fake, real)
    all_phv.append(phv_list)

all_phv = np.mean(np.array(all_phv), axis=0)
phv_str = ' '.join([f'{l}: {v:.4f}' for l, v in zip(layers, all_phv)])
print("[PHV]", phv_str)
phv_mean = float(np.mean(all_phv))
print(f"PHV_mean     : {phv_mean:.6f}")





# =========================================================
# PSNR / SSIM (CORRECT for RGB pathology)
# =========================================================
psnr_vals, ssim_vals = [], []
for s in tqdm(stems, desc="PSNR/SSIM"):
    fake = io.imread(os.path.join(pred_dir,s+img_ext))
    real = io.imread(os.path.join(targ_dir,s+img_ext))

    psnr_vals.append(peak_signal_noise_ratio(real, fake, data_range=255))
    ssim_vals.append(
        structural_similarity(
            real, fake,
            channel_axis=2,
            data_range=255
        )
    )

psnr = float(np.mean(psnr_vals))
ssim = float(np.mean(ssim_vals))

print(f"PSNR         : {psnr:.4f}")
print(f"SSIM         : {ssim:.4f}")



import lpips
from torchvision import transforms
from PIL import Image

# evaluate.py

# =========================================================
# LPIPS
# # =========================================================

lpips_model = lpips.LPIPS(net='alex').to(device)
tf_lpips = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

lpips_scores = []
print("[INFO] Computing LPIPS...")
for s in tqdm(stems, desc="LPIPS"):
    fake = tf_lpips(Image.open(os.path.join(pred_dir, s + img_ext)).convert('RGB')).unsqueeze(0).to(device)
    real = tf_lpips(Image.open(os.path.join(targ_dir, s + img_ext)).convert('RGB')).unsqueeze(0).to(device)
    lpips_scores.append(lpips_model(fake, real).item())
lpips_mean = float(np.mean(lpips_scores))
print(f"LPIPS: {lpips_mean:.4f}")



from DISTS_pytorch import DISTS
import cv2
from scipy.spatial.distance import jensenshannon


# =========================================================
# DISTS
# =========================================================
dists_model = DISTS().to(device)
dists_vals = []

for s in tqdm(stems, desc="DISTS"):
    fake = to_tensor(io.imread(os.path.join(pred_dir,s+img_ext))).unsqueeze(0).to(device)
    real = to_tensor(io.imread(os.path.join(targ_dir,s+img_ext))).unsqueeze(0).to(device)
    dists_vals.append(dists_model(fake, real).item())

dists_mean = float(np.mean(dists_vals))

# =========================================================
# JSD (DAB channel)
# =========================================================
def dab_hist(folder, bins=256):
    hist = np.zeros(bins)
    for f in os.listdir(folder):
        img = cv2.cvtColor(cv2.imread(os.path.join(folder,f)), cv2.COLOR_BGR2RGB)
        dab = rgb2hed(img)[:,:,2]
        h,_ = np.histogram(dab.flatten(), bins=bins, range=(-0.3,1.5))
        hist += h
    return hist / hist.sum()

jsd = float(jensenshannon(dab_hist(pred_dir), dab_hist(targ_dir)))


print(f"DISTS        : {dists_mean:.4f}")
print(f"JSD (DAB)    : {jsd:.6f}")


# =========================================================
# IoD, Rc, Rp
# =========================================================
Rc_list, Dhat, O = [], [], []

for s in tqdm(stems, desc="IoD / Pearson"):
    he   = to_gray01(io.imread(os.path.join(he_dir,s+img_ext)))
    fake = to_gray01(io.imread(os.path.join(pred_dir,s+img_ext)))
    real = to_gray01(io.imread(os.path.join(targ_dir,s+img_ext)))

    Rc_list.append(pearson_corr(he, fake))
    Dhat.append(compute_IoD(io.imread(os.path.join(pred_dir,s+img_ext))))
    O.append(compute_IoD(io.imread(os.path.join(targ_dir,s+img_ext))))

Rc = float(np.mean(Rc_list))
Rp = float(np.corrcoef(Dhat, O)[0,1])
Ravg = 0.5 * (Rc + Rp)

IoD = abs(np.sum(Dhat) - np.sum(O)) * 1e-7
mIoD = np.mean(np.abs(np.array(Dhat) - np.array(O))) * 1e-7

print(f"Rc (content) : {Rc:.6f}")
print(f"Rp (pathol.) : {Rp:.6f}")
print(f"Ravg         : {Ravg:.6f}")
print(f"IoD ×1e−7    : {IoD:.6f}")
print(f"mIoD ×1e−7   : {mIoD:.6f}")

# =========================================================
# Save metrics
# =========================================================
out = "Metrics/test/metrics.txt"
with open(out, "w") as f:
    f.write(f"PHV: {all_phv}\n")
    f.write(f"PHV_mean: {phv_mean:.6f}\n")
    f.write(f"FID: {fid_value:.4f}\n")
    f.write(f"KID_mean: {kid_mean:.6f}  KID_std: {kid_std:.6f}\n")
    f.write(f"PSNR: {psnr:.4f}\n")
    f.write(f"SSIM: {ssim:.4f}\n")
    f.write(f"LPIPS: {lpips_mean:.4f}\n")
    f.write(f"DISTS: {dists_mean:.4f}\n")
    f.write(f"JSD: {jsd:.6f}\n")
    f.write(f"Rc: {Rc:.6f}\n")
    f.write(f"Rp: {Rp:.6f}\n")
    f.write(f"Ravg: {Ravg:.6f}\n")
    f.write(f"IoD_x1e-7: {IoD:.6f}\n")
    f.write(f"mIoD_x1e-7: {mIoD:.6f}\n")

print(f"[DONE] Metrics saved to {out}")






#####################################################################

# # evaluate.py
# import os
# import random
# import torch
# import numpy as np
# import cv2
# from skimage import io, color
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# from skimage.color import rgb2hed
# from scipy.spatial.distance import jensenshannon
# from tqdm import tqdm
# from torchvision.transforms.functional import to_tensor
# from torchvision import transforms
# from PIL import Image

# from pytorch_fid.inception import InceptionV3
# from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
# import lpips
# from DISTS_pytorch import DISTS

# from util.perceptual import PerceptualHashValue
# from util.kid_score import compute_kid   # <<< IMPORTANT (no external script)

# # =========================================================
# # Paths
# # =========================================================
# os.makedirs("Metrics/test", exist_ok=True)

# targ_dir = 'results/train/val_latest/images/real_B'
# pred_dir = 'results/train/val_latest/images/fake_B'


# # ----------------------------
# # Image list (shared stem names)
# # ----------------------------
# img_list = [f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# assert len(img_list) > 0, "No images found in pred_dir"
# img_ext = '.' + img_list[0].split('.')[-1]
# stems = [os.path.splitext(f)[0] for f in img_list]

# # Keep only stems that exist in all three folders (pred, real_B, real_A)
# def has_all(s):
#     return (os.path.exists(os.path.join(pred_dir, s + img_ext))
#             and os.path.exists(os.path.join(targ_dir, s + img_ext)))
#            # and os.path.exists(os.path.join(he_dir,   s + img_ext)))
# stems = [s for s in stems if has_all(s)]
# random.seed(0)
# random.shuffle(stems)
# print(f"[INFO] Evaluating {len(stems)} image pairs.")

# device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')


# # ----------------------------
# # Helper: grayscale + float in [0,1]
# # ----------------------------
# def to_gray01(img):
#     if img.ndim == 3:
#         # assume RGB
#         g = color.rgb2gray(img)
#     else:
#         g = img.astype(np.float32)
#     g = g.astype(np.float32)
#     if g.max() > 1.0:
#         g = g / 255.0
#     return np.clip(g, 0.0, 1.0)


# # =========================================================
# # Optical Density (OD)
# # =========================================================
# def convert_RGB_to_OD(I):
#     I = I.astype(np.float32)
#     I[I == 0] = 1
#     return np.maximum(-np.log(I / 255.0), 1e-6)

# def compute_IoD(rgb):
#     OD = convert_RGB_to_OD(rgb)
#     return np.sum(np.mean(OD, axis=2))

# # =========================================================
# # Helper
# # =========================================================
# def to_gray01(img):
#     if img.ndim == 3:
#         img = color.rgb2gray(img)
#     img = img.astype(np.float32)
#     if img.max() > 1:
#         img /= 255.0
#     return img

# def pearson_corr(a, b):
#     a = a.ravel()
#     b = b.ravel()
#     if a.std() == 0 or b.std() == 0:
#         return 0.0
#     return float(np.corrcoef(a, b)[0, 1])


# # # ----------------------------
# # # PHV
# # # ----------------------------
# #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # layers = ['layer_1', 'layer_2', 'layer_3', 'layer_4']
# # PHV = PerceptualHashValue(
# #     T=0.01, network='resnet50', layers=layers,
# #     resize=False, resize_mode='bilinear', instance_normalized=False
# # ).to(device)

# # all_phv = []
# # for s in tqdm(stems, desc="PHV"):
# #     fake = io.imread(os.path.join(pred_dir, s + img_ext))
# #     real = io.imread(os.path.join(targ_dir, s + img_ext))
# #     fake = to_tensor(fake).to(device)
# #     real = to_tensor(real).to(device)
# #     phv_list = PHV(fake, real)
# #     all_phv.append(phv_list)

# # all_phv = np.mean(np.array(all_phv), axis=0)
# # phv_str = ' '.join([f'{l}: {v:.4f}' for l, v in zip(layers, all_phv)])
# # print("[PHV]", phv_str)
# # #phv_vals = np.mean(np.stack(all_phv), axis=0)
# # phv_mean = float(np.mean(all_phv))
# # print(f"PHV_mean     : {phv_mean:.6f}")


# # =========================================================
# # FID 
# # =========================================================

# num_avail_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
# num_workers = min(num_avail_cpus or 4, 8)


# real_paths = [os.path.join(targ_dir, s + img_ext) for s in stems]
# fake_paths = [os.path.join(pred_dir, s + img_ext) for s in stems]

# dims = 2048
# block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
# model = InceptionV3([block_idx]).to(device)

# m1, s1 = calculate_activation_statistics(real_paths, model, batch_size=10, dims=dims,
#                                          device=device, num_workers=num_workers)
# m2, s2 = calculate_activation_statistics(fake_paths, model, batch_size=10, dims=dims,
#                                          device=device, num_workers=num_workers)
# fid_value = calculate_frechet_distance(m1, s1, m2, s2)
# print(f'FID: {fid_value:.2f}')

# # =========================================================
# # KID (no external script)
# # =========================================================
# kid_mean, kid_std = compute_kid(targ_dir,pred_dir,batch_size=50,dims=2048,device='cuda')
# print(f"FID          : {fid_value:.4f}")
# print(f"KID_mean: {kid_mean:.6f} | KID_std: {kid_std:.6f}")

# # print(f"KID_mean     : {kid_mean:.6f}")
# # print(f"KID_std      : {kid_std:.6f}")
# print(f"KID_mean x1000 : {kid_mean * 1000:.6f}")



# # =========================================================
# # PSNR / SSIM (CORRECT for RGB pathology)
# # =========================================================
# psnr_vals, ssim_vals = [], []
# for s in tqdm(stems, desc="PSNR/SSIM"):
#     fake = io.imread(os.path.join(pred_dir,s+img_ext))
#     real = io.imread(os.path.join(targ_dir,s+img_ext))

#     psnr_vals.append(peak_signal_noise_ratio(real, fake, data_range=255))
#     ssim_vals.append(
#         structural_similarity(
#             real, fake,
#             channel_axis=2,
#             data_range=255
#         )
#     )

# psnr = float(np.mean(psnr_vals))
# ssim = float(np.mean(ssim_vals))

# print(f"PSNR         : {psnr:.4f}")
# print(f"SSIM         : {ssim:.4f}")


# # =========================================================
# # LPIPS
# # =========================================================
# lpips_model = lpips.LPIPS(net='alex').to(device)
# tf_lpips = transforms.Compose([
#     transforms.Resize((256,256)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,)*3,(0.5,)*3)
# ])

# lpips_vals = []
# for s in tqdm(stems, desc="LPIPS"):
#     fake = tf_lpips(Image.open(os.path.join(pred_dir,s+img_ext))).unsqueeze(0).to(device)
#     real = tf_lpips(Image.open(os.path.join(targ_dir,s+img_ext))).unsqueeze(0).to(device)
#     lpips_vals.append(lpips_model(fake, real).item())

# lpips_mean = float(np.mean(lpips_vals))
# print(f"LPIPS        : {lpips_mean:.4f}")

# # =========================================================
# # DISTS
# # =========================================================
# dists_model = DISTS().to(device)
# dists_vals = []

# for s in tqdm(stems, desc="DISTS"):
#     fake = to_tensor(io.imread(os.path.join(pred_dir,s+img_ext))).unsqueeze(0).to(device)
#     real = to_tensor(io.imread(os.path.join(targ_dir,s+img_ext))).unsqueeze(0).to(device)
#     dists_vals.append(dists_model(fake, real).item())

# dists_mean = float(np.mean(dists_vals))

# # =========================================================
# # JSD (DAB channel)
# # =========================================================
# def dab_hist(folder, bins=256):
#     hist = np.zeros(bins)
#     for f in os.listdir(folder):
#         img = cv2.cvtColor(cv2.imread(os.path.join(folder,f)), cv2.COLOR_BGR2RGB)
#         dab = rgb2hed(img)[:,:,2]
#         h,_ = np.histogram(dab.flatten(), bins=bins, range=(-0.3,1.5))
#         hist += h
#     return hist / hist.sum()

# jsd = float(jensenshannon(dab_hist(pred_dir), dab_hist(targ_dir)))


# print(f"DISTS        : {dists_mean:.4f}")
# print(f"JSD (DAB)    : {jsd:.6f}")


# # =========================================================
# # IoD, Rc, Rp
# # =========================================================
# Rc_list, Dhat, O = [], [], []

# for s in tqdm(stems, desc="IoD / Pearson"):
#     he   = to_gray01(io.imread(os.path.join(he_dir,s+img_ext)))
#     fake = to_gray01(io.imread(os.path.join(pred_dir,s+img_ext)))
#     real = to_gray01(io.imread(os.path.join(targ_dir,s+img_ext)))

#     Rc_list.append(pearson_corr(he, fake))
#     Dhat.append(compute_IoD(io.imread(os.path.join(pred_dir,s+img_ext))))
#     O.append(compute_IoD(io.imread(os.path.join(targ_dir,s+img_ext))))

# Rc = float(np.mean(Rc_list))
# Rp = float(np.corrcoef(Dhat, O)[0,1])
# Ravg = 0.5 * (Rc + Rp)

# IoD = abs(np.sum(Dhat) - np.sum(O)) * 1e-7
# mIoD = np.mean(np.abs(np.array(Dhat) - np.array(O))) * 1e-7

# print(f"Rc (content) : {Rc:.6f}")
# print(f"Rp (pathol.) : {Rp:.6f}")
# print(f"Ravg         : {Ravg:.6f}")
# print(f"IoD ×1e−7    : {IoD:.6f}")
# print(f"mIoD ×1e−7   : {mIoD:.6f}")

# # =========================================================
# # Save metrics
# # =========================================================
# out = "Metrics/test/metrics.txt"
# with open(out, "w") as f:
#     f.write(f"PHV: {phv_vals}\n")
#     f.write(f"PHV_mean: {phv_mean:.6f}\n")
#     f.write(f"FID: {fid:.4f}\n")
#     f.write(f"KID_mean: {kid_mean:.6f}  KID_std: {kid_std:.6f}\n")
#     f.write(f"PSNR: {psnr:.4f}\n")
#     f.write(f"SSIM: {ssim:.4f}\n")
#     f.write(f"LPIPS: {lpips_mean:.4f}\n")
#     f.write(f"DISTS: {dists_mean:.4f}\n")
#     f.write(f"JSD: {jsd:.6f}\n")
#     f.write(f"Rc: {Rc:.6f}\n")
#     f.write(f"Rp: {Rp:.6f}\n")
#     f.write(f"Ravg: {Ravg:.6f}\n")
#     f.write(f"IoD_x1e-7: {IoD:.6f}\n")
#     f.write(f"mIoD_x1e-7: {mIoD:.6f}\n")

# print(f"[DONE] Metrics saved to {out}")
