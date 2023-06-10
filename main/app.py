from flask import Flask, request, jsonify, make_response, g, send_file
import sys
from PIL import Image, ImageOps
import numpy as np
from rembg import remove
from predict_pose import generate_pose_keypoints
import os
import time

sys.path.append('U-2-Net')
import u2net_load, u2net_run

app = Flask(__name__)


@app.before_request
def load_variable():
    # preprocessing
    # import U-2-Net.u2net_run
    g.u2net = u2net_load.model(model_name='u2netp')


def composite_background(person_image_path, tryon_image_path, img_mask):
    from PIL import Image, ImageOps
    import numpy as np
    """Put background back on the person image after tryon."""
    person = np.array(Image.open(person_image_path))
    # tryon image
    tryon = np.array(Image.open(tryon_image_path))
    # persom image mask from rembg
    p_mask = np.array(img_mask)
    # make binary mask
    p_mask = np.where(p_mask > 0, 1, 0)
    # invert mask
    p_mask_inv = np.logical_not(p_mask)
    # make bg without person
    background = person * np.stack((p_mask_inv, p_mask_inv, p_mask_inv), axis=2)
    # make tryon image without background
    tryon_nobg = tryon * np.stack((p_mask, p_mask, p_mask), axis=2)
    tryon_nobg = tryon_nobg.astype("uint8")
    # composite
    tryon_with_bg = np.add(tryon_nobg, background)
    tryon_with_bg_pil = Image.fromarray(np.uint8(tryon_with_bg)).convert('RGB')
    tryon_with_bg_pil.save("results/test/try-on/tryon_with_bg.png")


@app.route('/hello')
def hello():
    # preprocessing
    from PIL import Image, ImageOps
    sys.path.append('U-2-Net')
    import u2net_load, u2net_run

    cloth_name = 'cloth.png'
    cloth_path = os.path.join('inputs/cloth', sorted(os.listdir('inputs/cloth'))[0])
    cloth = Image.open(cloth_path)

    # Resize cloth image
    cloth = ImageOps.fit(cloth, (192, 256), Image.BICUBIC).convert("RGB")

    # Save resized cloth image
    cloth.save(os.path.join('Data_preprocessing/test_color', cloth_name))

    # 1. Get binary mask for clothing image
    u2net_run.infer(g.u2net, 'Data_preprocessing/test_color', 'Data_preprocessing/test_edge')

    import time

    start_time = time.time()

    # Remove background from person image
    remove_bg = False
    # Person image
    img_name = 'person.png'
    img_path = os.path.join('inputs/img', sorted(os.listdir('inputs/img'))[0])
    img = Image.open(img_path)
    if remove_bg:
        # Remove background
        img = remove(img, alpha_matting=True, alpha_matting_erode_size=15)
        print("Removing background from person image..")
    img = ImageOps.fit(img, (192, 256), Image.BICUBIC).convert("RGB")
    # Get binary from person image
    img_mask = remove(img, alpha_matting=True, alpha_matting_erode_size=15, only_mask=True)
    img_path = os.path.join('Data_preprocessing/test_img', img_name)
    img.save(img_path)
    resize_time = time.time()
    print('Resized image in {}s'.format(resize_time - start_time))

    # 2. Get parsed person image (test_label), uses person image
    import subprocess
    import cv2

    command = ['python3', 'Self-Correction-Human-Parsing-CPU/schp_utils/simple_extractor.py',
               '--dataset', 'lip',
               '--model-restore', 'lip_final.pth',
               '--input-dir', 'Data_preprocessing/test_img',
               '--output-dir', 'Data_preprocessing/test_label']

    # start a new Python process and execute the script
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # wait for the process to complete and capture the output
    stdout, stderr = process.communicate()
    output = stdout.decode() + stderr.decode()
    print(output)
    # subprocess.run(cmd, capture_output=True)

    npy_path = "Data_preprocessing/temp/person.npy"
    logits = np.load(npy_path).astype(int)
    img = np.argmax(logits, axis=2)
    img = np.where(img < 10, img - 1, img - 2)
    img = np.where(img == -1, 0, img)
    cv2.imwrite('Data_preprocessing/test_label/person.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # plt.imshow(img)
    parse_time = time.time()
    print('Parsing generated in {}s'.format(parse_time - resize_time))
    from PIL import Image

    # Open the palette image
    # palette_image = Image.open('Data_preprocessing/test_label/person.png')

    # Convert to grayscale
    # grayscale_image = palette_image.convert('L')

    # Save the grayscale image
    # grayscale_image.save('Data_preprocessing/test_label/person.png')

    # 3. Get pose map from person image
    pose_path = os.path.join('Data_preprocessing/test_pose', img_name.replace('.png', '_keypoints.json'))
    generate_pose_keypoints(img_path, pose_path)
    pose_time = time.time()
    print('Pose map generated in {}s'.format(pose_time - parse_time))

    cmd = ['rm', '-rf', 'Data_preprocessing/test_pairs.txt']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode == 0:
        print("File deleted successfully.")
    else:
        print("Failed to delete the file.")

    # Format: person, cloth image
    with open('Data_preprocessing/test_pairs.txt', 'w') as f:
        f.write('person.png cloth.png')

    command = ['python', 'test.py', '--name', 'fifa_viton']
    # start a new Python process and execute the script
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # wait for the process to complete and capture the output
    stdout, stderr = process.communicate()
    output = stdout.decode() + stderr.decode()
    print(output)

    composite_background('Data_preprocessing/test_img/person.png',
                         'results/test/try-on/person.png', img_mask)

    return jsonify({'message': 'Hello, World!'})


@app.route('/process-cloth-image', methods=['POST'])
def process_cloth_image():
    cloth_name = 'cloth.png'
    # check if image file is present in the request
    if 'image' not in request.files:
        return make_response(jsonify({'error': 'No image file in the request'}), 400)

    # read the image file from the request
    image_file = request.files['image']

    # load the image using PIL library
    cloth = Image.open(image_file)

    # Resize cloth image
    cloth = ImageOps.fit(cloth, (192, 256), Image.BICUBIC).convert("RGB")

    # Save resized cloth image
    cloth.save(os.path.join('Data_preprocessing/test_color', cloth_name))

    # 1. Get binary mask for clothing image
    u2net_run.infer(g.u2net, 'Data_preprocessing/test_color', 'Data_preprocessing/test_edge')

    # Define the file paths for the images
    image2_path = 'Data_preprocessing/test_edge/% s' % cloth_name

    response = send_file(
        image2_path,
        # attachment_filename='image2.png',
        mimetype='image/jpeg'
    )
    response.headers['Content-Disposition'] = 'attachment; filename=image2.png'

    return response


@app.route('/process-person-image', methods=['POST'])
def process_person_image():
    from PIL import Image, ImageOps

    start_time = time.time()

    # Remove background from person image
    remove_bg = False
    # Person image
    img_name = 'person.png'
    # img_path = os.path.join('inputs/img', sorted(os.listdir('inputs/img'))[0])
    if 'image' not in request.files:
        return make_response(jsonify({'error': 'No image file in the request'}), 400)

        # read the image file from the request
    image_file = request.files['image']

    # load the image using PIL library
    img = Image.open(image_file)

    if remove_bg:
        # Remove background
        img = remove(img, alpha_matting=True, alpha_matting_erode_size=15)
        print("Removing background from person image..")
    img = ImageOps.fit(img, (192, 256), Image.BICUBIC).convert("RGB")
    # Get binary from person image
    img_mask = remove(img, alpha_matting=True, alpha_matting_erode_size=15, only_mask=True)
    img_path = os.path.join('Data_preprocessing/test_img', img_name)
    img.save(img_path)
    resize_time = time.time()
    print('Resized image in {}s'.format(resize_time - start_time))

    # 2. Get parsed person image (test_label), uses person image
    import subprocess
    import cv2

    command = ['python3', 'Self-Correction-Human-Parsing-CPU/schp_utils/simple_extractor.py',
               '--dataset', 'lip',
               '--model-restore', 'lip_final.pth',
               '--input-dir', 'Data_preprocessing/test_img',
               '--output-dir', 'Data_preprocessing/test_label']

    # start a new Python process and execute the script
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # wait for the process to complete and capture the output
    stdout, stderr = process.communicate()
    output = stdout.decode() + stderr.decode()
    print(output)

    npy_path = "Data_preprocessing/temp/person.npy"
    logits = np.load(npy_path).astype(int)
    img = np.argmax(logits, axis=2)
    img = np.where(img < 10, img - 1, img - 2)
    img = np.where(img == -1, 0, img)
    cv2.imwrite('Data_preprocessing/test_label/person.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    parse_time = time.time()
    print('Parsing generated in {}s'.format(parse_time - resize_time))
    from PIL import Image
    # 3. Get pose map from person image
    pose_path = os.path.join('Data_preprocessing/test_pose', img_name.replace('.png', '_keypoints.json'))
    generate_pose_keypoints(img_path, pose_path)
    pose_time = time.time()
    print('Pose map generated in {}s'.format(pose_time - parse_time))

    # Define the file paths for the images
    image2_path = 'Data_preprocessing/test_label/% s' % img_name

    # with open(pose_path, 'rb') as f:
    #     file_data = f.read()
    import io
    import zipfile

    # Return the file as a response
    # create a in-memory zip file
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w') as zf:
        # add files to the zip file
        zf.write(pose_path, 'file1.json')
        zf.write(image2_path, 'file2.png')

    # reset the file pointer to the beginning of the zip file
    mem.seek(0)

    # send the zip file to the client
    return send_file(mem, download_name='files.zip', as_attachment=True)

    # response = send_file(
    #     image2_path,
    #     # attachment_filename='image2.png',
    #     mimetype='image/jpeg'
    # )

    # response.headers['Content-Disposition'] = 'attachment; filename=image2.png'
    # image_data_response = make_response(jsonify({'image_data': image_data}))
    #
    # response2 = send_file(
    #     image2_path,
    #     # attachment_filename='image2.png',
    #     mimetype='image/jpeg'
    # )
    # response2.headers['Content-Disposition'] = 'attachment; filename=image2.png'
    #
    # return response, response2


@app.route('/virtual-tryon', methods=['POST'])
def virtual_tryon():
    from PIL import Image, ImageOps

    start_time = time.time()

    person = request.files.get('person')
    person_label = request.files.get('person_label')
    pose_keypoints_data = request.files.get('pose_keypoints_data')
    edge_image_data = request.files.get('edge_image_data')
    cloth_image_data = request.files.get('cloth_image_data')

    # person.save(os.path.join('Data_preprocessing/test_img', "person.png"))
    person.save(os.path.join('inputs/img', "person.png"))
    person_label.save(os.path.join('Data_preprocessing/test_label', "person.png"))
    pose_keypoints_data.save(os.path.join('Data_preprocessing/test_pose', "person_keypoints.json"))
    edge_image_data.save(os.path.join('Data_preprocessing/test_edge', "cloth.png"))
    # cloth_image_data.save(os.path.join('Data_preprocessing/test_color', "cloth.png"))
    cloth_image_data.save(os.path.join('inputs/cloth', "cloth.png"))

    cloth_name = 'cloth.png'
    cloth_path = os.path.join('inputs/cloth', sorted(os.listdir('inputs/cloth'))[0])
    cloth = Image.open(cloth_path)

    # Resize cloth image
    cloth = ImageOps.fit(cloth, (192, 256), Image.BICUBIC).convert("RGB")

    # Save resized cloth image
    cloth.save(os.path.join('Data_preprocessing/test_color', cloth_name))

    # if 'image' not in request.files:
    #     return make_response(jsonify({'error': 'No image file in the request'}), 400)

    # Format: person, cloth image
    with open('Data_preprocessing/test_pairs.txt', 'w') as f:
        f.write('person.png cloth.png')

    command = ['python', 'test.py', '--name', 'fifa_viton']

    import subprocess
    import cv2

    # start a new Python process and execute the script
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # wait for the process to complete and capture the output
    stdout, stderr = process.communicate()
    output = stdout.decode() + stderr.decode()
    print(output)
    img_name = 'person.png'
    # load the image using PIL library
    img = Image.open('inputs/img/person.png')
    img = ImageOps.fit(img, (192, 256), Image.BICUBIC).convert("RGB")
    # Get binary from person image
    img_mask = remove(img, alpha_matting=True, alpha_matting_erode_size=15, only_mask=True)

    img_path = os.path.join('Data_preprocessing/test_img', img_name)
    img.save(img_path)
    resize_time = time.time()
    print('Resized image in {}s'.format(resize_time - start_time))

    composite_background('Data_preprocessing/test_img/person.png',
                         'results/test/try-on/person.png', img_mask)

    # cmd = ['rm', '-rf', 'inputs/image/test_pairs.txt']
    # result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    response2 = send_file(
        'results/test/try-on/tryon_with_bg.png',
        # attachment_filename='image2.png',
        mimetype='image/jpeg'
    )

    response2.headers['Content-Disposition'] = 'attachment; filename=image2.png'

    return response2


if __name__ == '__main__':
    app.run(debug=True)
