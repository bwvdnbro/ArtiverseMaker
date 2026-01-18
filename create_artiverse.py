#!/usr/bin/env python3

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl

if __name__ == "__main__":
    np.random.seed(42)

    x = np.linspace(0.0, 10.0, 1000)
    y = np.linspace(0.0, 10.0, 1000)
    xs, ys = np.meshgrid(x, y)

    mask = np.zeros(xs.shape, dtype=bool)

    # face
    r2 = (xs - 5.0) ** 2 + (ys - 5.0) ** 2
    mask[(r2 < 5.0**2) & (r2 >= 4.8**2)] = True
    # mouth
    r = np.sqrt(r2)
    angle = np.atan2((ys - 5.0) / r, (xs - 5.0) / r)
    mask[
        (r2 <= 3.0**2)
        & (r2 >= 2.7**2)
        & (angle > -0.75 * np.pi)
        & (angle < -0.25 * np.pi)
    ] = True

    # eyes
    r12 = (xs - 3.5) ** 2 + (ys - 7.0) ** 2
    r22 = (xs - 6.5) ** 2 + (ys - 7.0) ** 2
    mask[(r12 <= 0.5**2) | (r22 <= 0.5**2)] = True

    rate = mask.sum() / mask.size

    target = 100
    sample_size = int(target / rate)

    params = np.random.rand(sample_size, 2) * 10.0
    xbin = np.digitize(params[:, 0], x)
    ybin = np.digitize(params[:, 1], y)
    params = params[mask[ybin, xbin]]
    while len(params) < target:
        xy = np.random.rand(int(0.1 * target / rate), 2) * 10.0
        xbin = np.digitize(xy[:, 0], x)
        ybin = np.digitize(xy[:, 1], y)
        params = np.append(params, xy[mask[ybin, xbin]], axis=0)
    params = params[:target]

    params[:, 0] = 1.0 + 0.9 * params[:, 0]
    params[:, 1] = 1.0 + 0.01 * params[:, 1]

    pl.semilogy(params[:, 0], params[:, 1], ".")
    pl.savefig("params.png", dpi=300)
    pl.close()

    maxx = 10000.0
    coords = np.random.rand(target, 2) * maxx
    I0s = 1.0 + 0.1 * np.random.rand(target)

    imgcoord = np.linspace(0.1 * maxx, 0.9 * maxx, 2048)
    imgx, imgy = np.meshgrid(imgcoord, imgcoord)
    img = np.zeros(imgx.shape)
    print(img.shape)
    i = 0
    for (x, y), I0, (n, rs) in zip(coords, I0s, params):
        r = np.sqrt((imgx - x) ** 2 + (imgy - y) ** 2)
        I = I0 * np.exp(-((r / rs) ** (1.0 / n)))
        print(I.max())
        img += I
        print(i)
        i += 1

    np.savez_compressed(
        "full_sample.npz",
        coords=coords,
        I0s=I0s,
        ns=params[:, 0],
        rss=params[:, 1],
        img=img,
    )

    pl.imshow(img)
    pl.savefig("test.png", dpi=300)
