import os
import shutil

from PIL import Image, ImageEnhance


def image_enhance(image, bright, contrast, color, sharpness):
    # 亮度调整
    brightEnhancer = ImageEnhance.Brightness(image)
    image = brightEnhancer.enhance(bright)
    # 对比度调整
    contrastEnhancer = ImageEnhance.Contrast(image)
    image = contrastEnhancer.enhance(contrast)
    # 饱和度调整
    colorEnhancer = ImageEnhance.Color(image)
    image = colorEnhancer.enhance(color)
    # 清晰度调整
    SharpnessEnhancer = ImageEnhance.Sharpness(image)
    image = SharpnessEnhancer.enhance(sharpness)

    return image


isp_coeff_list = [0.8, 0.9, 1.1, 1.2]
for file in os.listdir("images"):
    image = Image.open(f"images/{file}")
    i = 1
    for bright in isp_coeff_list:
        for contrast in isp_coeff_list:
            for color in isp_coeff_list:
                for sharpness in isp_coeff_list:
                    new_image = image_enhance(image, bright, contrast, color, sharpness)
                    new_image.save(f"images/{file[:-4]}_{i:04d}.jpg")
                    shutil.copyfile(f"labels/{file[:-4]}.txt", f"labels/{file[:-4]}_{i:04d}.txt")
                    i += 1
