import os
import zipfile
from PIL import Image
from PIL.Image import Resampling

from src.util import download_file, run_script_with_args


def upscale(input_dir, output_dir, scale_factor=3, width=512, height=512):
    dir = "../resource/model/esrgan"
    script_path = "../resource/model/esrgan/realesrgan-ncnn-vulkan"

    if not os.path.isfile(script_path):
        zip_path = "../resource/model/esrgan.zip"
        os.makedirs(dir, exist_ok=True)

        try:
            download_file(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
                destination=zip_path
            )
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dir)
                contents = zip_ref.namelist()
                if len(contents) == 1:
                    raise AssertionError("Zip content is wrong.")
        except zipfile.BadZipFile as e:
            print(f"Error unpacking file: {e}")
            return
        except Exception as e:
            print(f"Unexpected error during unpacking: {e}")
            return

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        if os.path.isfile(input_path):
            output_path = os.path.join(output_dir, filename)
            print(f"{input_path} -> {output_path}")
            run_script_with_args(script_path, [
                '-i', input_path,
                '-o', output_path,
                '-n', "realesrgan-x4plus",
                # '-n', 'realesrnet-x4plus',
                '-s', f"{scale_factor}",
            ])
            try:
                with Image.open(output_path) as img:
                    img.load()
                    img = img.resize((width, height), Resampling.LANCZOS)
                    img.save(output_path)
            except Exception as e:
                print(f"Failed to resize {output_path}: {e}")


# def upscale(input_dir, output_dir):
#     model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
#     model_name = 'RealESRGAN_x4plus'
#     suffix = 'out'
#     netscale = 4
#     outscale = 4
#     tile = 0
#     tile_pad = 10
#     pre_pad = 0
#     fp32 = False
#     face_enhance = False
#     denoise_strength = 0.5
#     file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
#
#     model_path = os.path.join('weights', model_name + '.pth')
#     if not os.path.isfile(model_path):
#         ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#         for url in file_url:
#             # model_path will be updated
#             model_path = load_file_from_url(
#                 url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
#
#     # use dni to control the denoise strength
#     dni_weight = None
#
#     # restorer
#     upsampler = RealESRGANer(
#         scale=netscale,
#         model_path=model_path,
#         dni_weight=dni_weight,
#         model=model,
#         tile=tile,
#         tile_pad=tile_pad,
#         pre_pad=pre_pad,
#         half=not fp32,
#         gpu_id=None)
#
#     if face_enhance:  # Use GFPGAN for face enhancement
#         from gfpgan import GFPGANer
#         face_enhancer = GFPGANer(
#             model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
#             upscale=outscale,
#             arch='clean',
#             channel_multiplier=2,
#             bg_upsampler=upsampler)
#     os.makedirs(output_dir, exist_ok=True)
#
#     if os.path.isfile(input_dir):
#         paths = [input_dir]
#     else:
#         paths = sorted(glob.glob(os.path.join(input_dir, '*')))
#
#     for idx, path in enumerate(paths):
#         imgname, extension = os.path.splitext(os.path.basename(path))
#         print('Testing', idx, imgname)
#
#         img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#         if len(img.shape) == 3 and img.shape[2] == 4:
#             img_mode = 'RGBA'
#         else:
#             img_mode = None
#
#         try:
#             if face_enhance:
#                 _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
#             else:
#                 output, _ = upsampler.enhance(img, outscale=outscale)
#         except RuntimeError as error:
#             print('Error', error)
#         else:
#             extension = extension[1:]
#
#             if img_mode == 'RGBA':  # RGBA images should be saved in png format
#                 extension = 'png'
#             if suffix == '':
#                 save_path = os.path.join(output_dir, f'{imgname}.{extension}')
#             else:
#                 save_path = os.path.join(output_dir, f'{imgname}_{suffix}.{extension}')
#             cv2.imwrite(save_path, output)


if __name__ == "__main__":
    input_dir = "../output/cropped"
    output_dir = "../output/upscaled"
    upscale(input_dir, output_dir)
