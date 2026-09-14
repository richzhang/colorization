# Gray-to-Glow — Colorful Image Colorization

This project provides automatic colorization of grayscale images using pretrained models from:
- "Colorful Image Colorization" (ECCV 2016)
- "Real-Time User-Guided Image Colorization with Learned Deep Priors" (SIGGRAPH 2017)

The original research and pretrained models were adapted to a small PyTorch-based demo. This repository now includes a simple web UI (Flask) so you can run a local site, upload images, and get colorized outputs.

---

## What's included

- The original command-line demo (`demo_release.py`) for single-image colorization.
- Two pretrained colorizers (ECCV16 and SIGGRAPH17) in `colorizers/`.
- A Flask server (`server.py`) and a lightweight web UI (in `templates/` and `static/`) that runs on port 8080 by default.
- Utility functions for preprocessing/postprocessing images in `colorizers/util.py`.

---

## New: Web UI (local)

Start the project as a web application and use a browser to upload images and see results.

1. Install dependencies (recommended in a virtualenv). Note: install a compatible `torch` build for your platform/GPU from https://pytorch.org before or instead of the generic requirements, if needed.

   ```
   pip install -r requirements.txt
   ```

   If you need a specific PyTorch (CUDA) wheel, install it following the official instructions at https://pytorch.org.

2. Run the server:

   ```
   python server.py
   ```

3. Open a browser and go to:

   ```
   http://localhost:8080
   ```

UI behavior:
- Upload a grayscale or color image and click `Colorize`. For colored uploads, the
  original image is used as the reference for evaluation after the model receives
  its grayscale lightness channel.
- A "Processing..." loader appears while the models run; when finished a centered slideshow is shown.
- The slideshow shows three images (Original, ECCV16, SIGGRAPH17). ECCV16 is centered by default. Use the Prev/Next buttons or thumbnails to switch.
- Outputs are written to `imgs_out/`.
- Each result includes mean CIEDE2000 color difference, MSE, PSNR, SSIM, and a
  similarity score from 0 to 100. CIEDE2000 and MSE are error metrics where
  lower is better. PSNR and SSIM are quality metrics where higher is better.
  The displayed similarity score is `max(0, 100 - mean CIEDE2000)`, so an exact
  color match scores 100. PSNR is reported in dB and uses RGB values normalized
  to the 0-1 range.

File naming (new behavior):
- If you upload `photo.jpg`, outputs are saved as:
  - `imgs_out/photo_eccv16.png`
  - `imgs_out/photo_siggraph17.png`
  - `imgs_out/photo_orig.png`
- Existing files with the same names will be overwritten by new results.

GPU:
- The demo runs on CPU by default. To use GPU, ensure you have a CUDA-enabled PyTorch installed and set `USE_GPU = True` at the top of `server.py`.

Model download:
- The pretrained weights are automatically downloaded the first time the models are loaded (requires internet).

---

## Command-line usage (unchanged)

You can still use the original demo script from the command-line:

```
python demo_release.py -i imgs/ansel_adams3.jpg
```

This will produce outputs in `imgs_out/` as before.

---

## Dependencies

The main dependencies used by the web UI and demo are:
- `torch` — model runtime (install correct wheel for CPU/CUDA)
- `scikit-image` — color space conversions
- `numpy` — array handling
- `matplotlib` — saving images (used for writing outputs)
- `Pillow` — basic image IO
- `Flask` — lightweight web server and UI

See `requirements.txt` for the pinned list. Replace `torch` install with the appropriate command from https://pytorch.org if you want CUDA support.

---

## Notes & troubleshooting

- If the server prints errors about downloading model weights, ensure outbound internet access is available.
- The models expect images to be in RGB; grayscale images are supported (they will be converted internally).
- Large images are resized to 256×256 for model inference and the predicted color channels are upsampled back to the original resolution.

---

## Citation

If you use these models in your research, please cite the original papers:

```
@inproceedings{zhang2016colorful,
  title={Colorful Image Colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle={ECCV},
  year={2016}
}

@article{zhang2017real,
  title={Real-Time User-Guided Image Colorization with Learned Deep Priors},
  author={Zhang, Richard and Zhu, Jun-Yan and Isola, Phillip and Geng, Xinyang and Lin, Angela S and Yu, Tianhe and Efros, Alexei A},
  journal={ACM Transactions on Graphics (TOG)},
  volume={9},
  number={4},
  year={2017},
  publisher={ACM}
}
```

---

## Contact

This project is adapted from the original research by Richard Zhang et al. For questions regarding the models, see the original project page: http://richzhang.github.io/colorization/.

If you want additional UI features (fade transitions, keyboard navigation, server-side queuing), I can add them.
