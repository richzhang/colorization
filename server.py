from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
import os
import uuid
import traceback
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import torch

from colorizers.eccv16 import eccv16
from colorizers.siggraph17 import siggraph17
from colorizers.util import load_img, preprocess_img, postprocess_tens
from colorizers.evaluation import evaluate_colorization

# configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'imgs')
OUT_FOLDER = os.path.join(os.path.dirname(__file__), 'imgs_out')
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'bmp'}
USE_GPU = False

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUT_FOLDER'] = OUT_FOLDER

# ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUT_FOLDER'], exist_ok=True)

# load models once
print('Loading colorization models...')
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if USE_GPU and torch.cuda.is_available():
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()
print('Models loaded.')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def colorize_image(input_path, save_prefix):
    try:
        img = load_img(input_path)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        if USE_GPU and torch.cuda.is_available():
            tens_l_rs = tens_l_rs.cuda()

        img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
        out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
        out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

        out_e = os.path.join(app.config['OUT_FOLDER'], f"{save_prefix}_eccv16.png")
        out_s = os.path.join(app.config['OUT_FOLDER'], f"{save_prefix}_siggraph17.png")
        out_o = os.path.join(app.config['OUT_FOLDER'], f"{save_prefix}_orig.png")

        plt.imsave(out_e, out_img_eccv16)
        plt.imsave(out_s, out_img_siggraph17)
        plt.imsave(out_o, img)

        eccv_score = evaluate_colorization(img, out_img_eccv16)
        siggraph_score = evaluate_colorization(img, out_img_siggraph17)

        return {
            'original': os.path.basename(out_o),
            'eccv16': os.path.basename(out_e),
            'siggraph17': os.path.basename(out_s),
            'scores': {
                'eccv16': eccv_score,
                'siggraph17': siggraph_score,
            },
        }
    except Exception as e:
        traceback.print_exc()
        raise


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'no file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'no selected file'}), 400
    if file and allowed_file(file.filename):
        # save uploaded image using the original (secure) filename so outputs can use the same base name
        original_filename = secure_filename(file.filename)
        base_name, _ = os.path.splitext(original_filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(upload_path)

        try:
            out_prefix = base_name
            result = colorize_image(upload_path, out_prefix)
            # return URLs for the client to fetch
            return jsonify({
                'success': True,
                'orig_url': f"/output/{result['original']}",
                'eccv_url': f"/output/{result['eccv16']}",
                'sig_url': f"/output/{result['siggraph17']}",
                'scores': result['scores'],
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    else:
        return jsonify({'success': False, 'error': 'file type not allowed'}), 400


@app.route('/output/<path:filename>')
def output_file(filename):
    try:
        return send_from_directory(app.config['OUT_FOLDER'], filename)
    except Exception:
        abort(404)


if __name__ == '__main__':
    # app runs on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
