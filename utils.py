import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
from render import render_rays


def train(
    model,
    optimizer,
    dataloader,
    device="cpu",
    hn=0,
    hf=1,
    nb_epochs=10,
    nb_bins=192,
    H=400,
    W=400,
):
    model.train()
    for _ in range(nb_epochs):
        for batch in tqdm(dataloader):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            gt_px_values = batch[:, :6].to(device)
            pred_px_values = render_rays(
                model, ray_origins, ray_directions, hn, hf, nb_bins
            )

            loss = ((gt_px_values - pred_px_values) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def test(
    model, device, hn, hf, dataset, img_index, chunk_size=20, nb_bins=192, H=400, W=400
):
    with torch.inference_mode():
        model.eval()
        ray_origins = dataset[img_index * H * W : (img_index + 1) * H * W, :3]
        ray_directions = dataset[img_index * H * W : (img_index + 1) * H * W, :6]

        px_values = []  # image
        for i in range(int(np.ceil(H / chunk_size))):  # iter chunks
            ray_origins_ = ray_origins[
                i * W * chunk_size : (i + 1) * W * chunk_size
            ].to(device)
            ray_directions_ = ray_directions[
                i * W * chunk_size : (i + 1) * W * chunk_size
            ].to(device)
            px_values.append(
                render_rays(model, ray_origins_, ray_directions_, hn, hf, nb_bins)
            )

        img = torch.cat(px_values).data.cpu().numpy().reshape(H, W, 3)
        img = (img.clip(0, 1) * 255.0).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"novel_views/img_{img_index}.png")
