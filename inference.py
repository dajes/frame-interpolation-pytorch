import bisect
import os
from tqdm import tqdm
import torch
import numpy as np
import cv2

from util import load_image


def inference(model_path, img1, img2, save_path, gpu, inter_frames, fps, half):
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    img_batch_1, crop_region_1 = load_image(img1)
    img_batch_2, crop_region_2 = load_image(img2)

    img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
    img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2)

    if not half:
        model.float()

    if gpu and torch.cuda.is_available():
        if half:
            model = model.half()
        else:
            model.float()
        model = model.cuda()

    if save_path == 'img1 folder':
        save_path = os.path.join(os.path.split(img1)[0], 'output.mp4')

    results = [
        img_batch_1,
        img_batch_2
    ]

    idxes = [0, inter_frames + 1]
    remains = list(range(1, inter_frames + 1))

    splits = torch.linspace(0, 1, inter_frames + 2)

    for _ in tqdm(range(len(remains)), 'Generating in-between frames'):
        starts = splits[idxes[:-1]]
        ends = splits[idxes[1:]]
        distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
        matrix = torch.argmin(distances).item()
        start_i, step = np.unravel_index(matrix, distances.shape)
        end_i = start_i + 1

        x0 = results[start_i]
        x1 = results[end_i]

        if gpu and torch.cuda.is_available():
            if half:
                x0 = x0.half()
                x1 = x1.half()
            x0 = x0.cuda()
            x1 = x1.cuda()

        dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

        with torch.no_grad():
            prediction = model(x0, x1, dt)
        insert_position = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_position, remains[step])
        results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
        del remains[step]

    video_folder = os.path.split(save_path)[0]
    os.makedirs(video_folder, exist_ok=True)

    y1, x1, y2, x2 = crop_region_1
    frames = [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]

    w, h = frames[0].shape[1::-1]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)

    for frame in frames[1:][::-1]:
        writer.write(frame)

    writer.release()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test frame interpolator model')

    parser.add_argument('model_path', type=str, help='Path to the TorchScript model')
    parser.add_argument('img1', type=str, help='Path to the first image')
    parser.add_argument('img2', type=str, help='Path to the second image')

    parser.add_argument('--save_path', type=str, default='img1 folder', help='Path to save the interpolated frames')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--fp16', action='store_true', help='Use FP16')
    parser.add_argument('--frames', type=int, default=18, help='Number of frames to interpolate')
    parser.add_argument('--fps', type=int, default=10, help='FPS of the output video')

    args = parser.parse_args()

    inference(args.model_path, args.img1, args.img2, args.save_path, args.gpu, args.frames, args.fps, args.fp16)
