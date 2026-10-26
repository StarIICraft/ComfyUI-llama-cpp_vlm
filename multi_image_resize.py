from __future__ import annotations

import math
import re

import numpy as np
import torch
from PIL import Image

import comfy.model_management as mm


MODE_LIMIT = "统一百万像素限制 / Megapixel limit"
MODE_FORCE = "所有图像强制缩放到百万像素 / Force megapixel size"
MODE_MANUAL = "图1手动尺寸/边模式 / Image 1 manual size or edge"
MODE_PAD = "统一比例填充 / Pad to aspect ratio"

MODES = [MODE_LIMIT, MODE_FORCE, MODE_MANUAL, MODE_PAD]

RATIO_PRESETS = [
    "手动", "原图", "1:1", "2:3", "3:2", "3:4", "4:3",
    "4:5", "5:4", "9:16", "16:9", "21:9",
    "1:4", "4:1", "1:8", "8:1",
]

COLOR_PRESETS = {
    "黑色 / Black": (0, 0, 0),
    "白色 / White": (255, 255, 255),
    "灰色 / Gray": (128, 128, 128),
    "红色 / Red": (255, 0, 0),
    "绿色 / Green": (0, 255, 0),
    "蓝色 / Blue": (0, 0, 255),
    "黄色 / Yellow": (255, 255, 0),
}
CUSTOM_COLOR = "自定义 / Custom"

METHOD_LONG = "按长边保持比例 / Fit long edge"
METHOD_SHORT = "按短边保持比例 / Fit short edge"
METHOD_CUSTOM = "自定义宽高 / Custom width and height"
METHODS = [METHOD_LONG, METHOD_SHORT, METHOD_CUSTOM]

FIT_STRETCH = "拉伸 / Stretch"
FIT_CROP = "裁剪 / Crop"
FIT_PAD = "填充 / Pad"
FIT_MODES = [FIT_STRETCH, FIT_CROP, FIT_PAD]

RESAMPLING = getattr(Image, "Resampling", Image)

INTERPOLATIONS = {
    "Lanczos": RESAMPLING.LANCZOS,
    "双三次 / Bicubic": RESAMPLING.BICUBIC,
    "双线性 / Bilinear": RESAMPLING.BILINEAR,
    "邻近 / Nearest": RESAMPLING.NEAREST,
    "区域 / Box": RESAMPLING.BOX,
}
MASK_INTERPOLATION = RESAMPLING.BILINEAR

MAX_IMAGES = 8


def check_interrupt():
    mm.throw_exception_if_processing_interrupted()


def parse_ratio(value):
    value = str(value or "").strip()
    value = value.replace("：", ":").replace("/", ":")

    try:
        if ":" in value:
            parts = value.split(":")
            if len(parts) != 2:
                raise ValueError
            width, height = map(float, parts)
            if (
                not math.isfinite(width)
                or not math.isfinite(height)
                or width <= 0
                or height <= 0
            ):
                raise ValueError
            ratio = width / height
        else:
            ratio = float(value)

        if not math.isfinite(ratio) or not 0.01 <= ratio <= 100:
            raise ValueError

    except (ValueError, ZeroDivisionError, OverflowError):
        raise ValueError(
            "手动比例无效，请填写1:1、16:9或0.75；"
            "范围为1:100～100:1。\n"
            "Invalid ratio; use 1:1, 16:9, or 0.75, "
            "within 1:100–100:1."
        ) from None

    return ratio


def parse_color(value):
    value = str(value or "").strip()

    if re.fullmatch(r"#[0-9a-fA-F]{3}", value):
        return tuple(int(ch * 2, 16) for ch in value[1:])

    if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))

    if value.lower().startswith("rgb"):
        match = re.fullmatch(r"rgb\s*\((.*)\)", value, re.I)
        if not match:
            raise ValueError("RGB格式无效 / Invalid RGB format.")
        value = match.group(1)

    parts = value.replace("，", ",").split(",")

    if len(parts) == 3 and all(
        re.fullmatch(r"\s*\d{1,3}\s*", part)
        for part in parts
    ):
        color = tuple(int(part.strip()) for part in parts)
        if all(0 <= channel <= 255 for channel in color):
            return color

    raise ValueError(
        "自定义颜色请填写#000000、0,0,0或rgb(0,0,0)。\n"
        "Use #000000, 0,0,0, or rgb(0,0,0)."
    )


def validate_image(value, name):
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"{name}需要IMAGE张量 / IMAGE tensor required."
        )

    if value.ndim == 3:
        value = value.unsqueeze(0)

    if (
        value.ndim != 4
        or min(value.shape[:3]) <= 0
        or value.shape[-1] not in (1, 3, 4)
    ):
        raise ValueError(
            f"{name}需要有效的[B,H,W,C]图像，通道数为1、3、4。\n"
            "Expected [B,H,W,C] with 1, 3, or 4 channels."
        )

    return value


def tensor_to_pil(frame, background):
    array = frame.detach().cpu().float().numpy()
    array = np.nan_to_num(array, nan=0, posinf=1, neginf=0)
    array = (np.clip(array, 0, 1) * 255 + 0.5).astype(np.uint8)

    if array.shape[-1] == 1:
        return Image.fromarray(array[..., 0]).convert("RGB")

    if array.shape[-1] == 3:
        return Image.fromarray(array)

    rgba = Image.fromarray(array).convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (*background, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def prepare_masks(mask, batch_size):
    if mask is None:
        return None

    if not isinstance(mask, torch.Tensor):
        raise ValueError(
            "图1遮罩必须是MASK张量 / MASK tensor required."
        )

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)

    if (
        mask.ndim != 3
        or mask.shape[0] not in (1, batch_size)
        or min(mask.shape[1:]) <= 0
    ):
        raise ValueError(
            "遮罩需为[H,W]或[B,H,W]，批次为1或与图1相同。\n"
            "Mask batch must be 1 or match image 1."
        )

    return mask


def mask_to_pil(mask, index, size):
    if mask is None:
        return Image.new("L", size, 255)

    frame = mask[0 if mask.shape[0] == 1 else index]
    array = frame.detach().cpu().float().numpy()
    array = np.nan_to_num(array, nan=0, posinf=1, neginf=0)
    array = (np.clip(array, 0, 1) * 255 + 0.5).astype(np.uint8)

    result = Image.fromarray(array)
    if result.size != size:
        result = result.resize(size, MASK_INTERPOLATION)
    return result


def fit_area(width, height, area, upscale, divisor):
    """计算不超过目标面积的尺寸。"""
    area = int(area)

    if width <= 0 or height <= 0 or area <= 0:
        raise ValueError(
            "图像尺寸与目标面积必须大于0 / Invalid dimensions or area."
        )

    if divisor > 1 and divisor * divisor > area:
        raise ValueError(
            "目标面积小于整除数平方，请减小整除数。\n"
            "Target area is too small for the divisor."
        )

    scale = math.sqrt(area / (width * height))
    if not upscale:
        scale = min(1.0, scale)

    width_out = max(1, int(math.floor(width * scale)))
    height_out = max(1, int(math.floor(height * scale)))

    if width_out * height_out > area:
        if width_out >= height_out:
            width_out = max(1, area // height_out)
        else:
            height_out = max(1, area // width_out)

    if divisor > 1:
        width_out = max(
            divisor,
            width_out // divisor * divisor,
        )
        height_out = max(
            divisor,
            height_out // divisor * divisor,
        )

        if width_out * height_out > area:
            if width_out >= height_out:
                width_out = (
                    area // height_out // divisor * divisor
                )
            else:
                height_out = (
                    area // width_out // divisor * divisor
                )

        if width_out < divisor or height_out < divisor:
            raise ValueError(
                "无法同时满足面积和整除要求。\n"
                "Cannot satisfy both area and divisor."
            )

    return int(width_out), int(height_out)


def rounded_size(width, height, divisor):
    width = max(1, int(width))
    height = max(1, int(height))

    if divisor > 1:
        width = max(divisor, width // divisor * divisor)
        height = max(divisor, height // divisor * divisor)

    return width, height


def resize_pair(image, mask, size, interpolation):
    if image.size != size:
        image = image.resize(size, interpolation)

    if mask is not None and mask.size != size:
        mask = mask.resize(size, MASK_INTERPOLATION)

    return image, mask


def area_resize(
    image,
    mask,
    area,
    divisor,
    interpolation,
    force=False,
):
    # 未超过面积上限且已经满足整除要求时，保持原尺寸。
    already_aligned = (
        divisor <= 1
        or (
            image.width % divisor == 0
            and image.height % divisor == 0
        )
    )

    if (
        not force
        and image.width * image.height <= area
        and already_aligned
    ):
        return image, mask

    size = fit_area(
        image.width,
        image.height,
        area,
        force,
        divisor,
    )

    return resize_pair(
        image,
        mask,
        size,
        interpolation,
    )


def place_on_canvas(
    image,
    mask,
    canvas_size,
    content_size,
    color,
    interpolation,
):
    """居中填充；输入遮罩的新填充区域标为白色。"""
    image, mask = resize_pair(
        image,
        mask,
        content_size,
        interpolation,
    )

    width, height = canvas_size
    left = (width - content_size[0]) // 2
    top = (height - content_size[1]) // 2

    canvas = Image.new("RGB", canvas_size, color)
    canvas.paste(image, (left, top))

    output_mask = None
    if mask is not None:
        output_mask = Image.new("L", canvas_size, 255)
        output_mask.paste(mask, (left, top))

    padding = Image.new("L", canvas_size, 255)
    padding.paste(
        0,
        (
            left,
            top,
            left + content_size[0],
            top + content_size[1],
        ),
    )

    return canvas, output_mask, padding


def pad_to_ratio(
    image,
    mask,
    ratio,
    color,
    area,
    divisor,
    interpolation,
):
    source_width, source_height = image.size

    if source_width / source_height < ratio:
        virtual = (
            max(source_width, math.ceil(source_height * ratio)),
            source_height,
        )
    else:
        virtual = (
            source_width,
            max(source_height, math.ceil(source_width / ratio)),
        )

    if math.isclose(
        source_width / source_height,
        ratio,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        virtual = image.size

    size = fit_area(
        virtual[0],
        virtual[1],
        area,
        False,
        divisor,
    )

    # 直接映射到最终画布，避免创建巨大中间图像。
    scale = min(
        size[0] / source_width,
        size[1] / source_height,
    )
    content = (
        max(1, min(size[0], round(source_width * scale))),
        max(1, min(size[1], round(source_height * scale))),
    )

    return place_on_canvas(
        image,
        mask,
        size,
        content,
        color,
        interpolation,
    )


def manual_resize(
    image,
    mask,
    width,
    height,
    edge,
    method,
    fit,
    color,
    divisor,
    interpolation,
):
    source_width, source_height = image.size

    if method in (METHOD_LONG, METHOD_SHORT):
        denominator = (
            max(source_width, source_height)
            if method == METHOD_LONG
            else min(source_width, source_height)
        )
        scale = edge / denominator

        size = rounded_size(
            round(source_width * scale),
            round(source_height * scale),
            divisor,
        )

        image, mask = resize_pair(
            image,
            mask,
            size,
            interpolation,
        )
        return image, mask, Image.new("L", size, 0)

    size = rounded_size(width, height, divisor)
    width, height = size

    if fit == FIT_STRETCH:
        image, mask = resize_pair(
            image,
            mask,
            size,
            interpolation,
        )
        return image, mask, Image.new("L", size, 0)

    if fit == FIT_CROP:
        scale = max(
            width / source_width,
            height / source_height,
        )
        scaled = (
            max(width, math.ceil(source_width * scale)),
            max(height, math.ceil(source_height * scale)),
        )

        image, mask = resize_pair(
            image,
            mask,
            scaled,
            interpolation,
        )

        left = (scaled[0] - width) // 2
        top = (scaled[1] - height) // 2
        box = (left, top, left + width, top + height)

        image = image.crop(box)
        if mask is not None:
            mask = mask.crop(box)

        return image, mask, Image.new("L", size, 0)

    scale = min(
        width / source_width,
        height / source_height,
    )
    content = (
        max(1, min(width, round(source_width * scale))),
        max(1, min(height, round(source_height * scale))),
    )

    return place_on_canvas(
        image,
        mask,
        size,
        content,
        color,
        interpolation,
    )


def images_to_tensor(images):
    width, height = images[0].size
    result = torch.empty(
        (len(images), height, width, 3),
        dtype=torch.float32,
    )

    for index, image in enumerate(images):
        check_interrupt()
        result[index].copy_(torch.from_numpy(
            np.asarray(image, dtype=np.float32) / 255.0
        ))

    return result


def masks_to_tensor(masks):
    width, height = masks[0].size
    result = torch.empty(
        (len(masks), height, width),
        dtype=torch.float32,
    )

    for index, mask in enumerate(masks):
        check_interrupt()
        result[index].copy_(torch.from_numpy(
            np.asarray(mask, dtype=np.float32) / 255.0
        ))

    return result


def combine_preview(outputs, color):
    images = []

    for output in outputs:
        if output is None:
            continue

        for frame in output:
            array = frame.detach().cpu().float().numpy()
            array = np.nan_to_num(array, nan=0, posinf=1, neginf=0)
            array = (np.clip(array, 0, 1) * 255 + 0.5).astype(np.uint8)
            images.append(Image.fromarray(array, "RGB"))

    if not images:
        return torch.empty((0, 1, 1, 3), dtype=torch.float32)

    width = max(image.width for image in images)
    height = max(image.height for image in images)
    canvas_images = []

    for image in images:
        canvas = Image.new("RGB", (width, height), color)
        left = (width - image.width) // 2
        top = (height - image.height) // 2
        canvas.paste(image, (left, top))
        canvas_images.append(canvas)

    return images_to_tensor(canvas_images)


class GoohaiMultiImageResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "language": (
                    ["中文", "English"],
                    {"default": "中文"},
                ),
                "resize_mode": (
                    MODES,
                    {"default": MODE_LIMIT},
                ),
                "megapixels": ("FLOAT", {
                    "default": 2.5,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.1,
                }),
                "area_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.05,
                }),
                "color_preset": (
                    list(COLOR_PRESETS) + [CUSTOM_COLOR],
                    {"default": "灰色 / Gray"},
                ),
                "fill_color": ("STRING", {
                    "default": "0,0,0",
                }),
                "ratio_preset": (
                    RATIO_PRESETS,
                    {"default": "原图"},
                ),
                "aspect_ratio": ("STRING", {
                    "default": "1:1",
                }),
                "manual_width": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 32768,
                }),
                "manual_height": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 32768,
                }),
                "edge_length": ("INT", {
                    "default": 2048,
                    "min": 1,
                    "max": 32768,
                }),
                "crop_mode": (
                    FIT_MODES,
                    {"default": FIT_PAD},
                ),
                "resize_method": (
                    METHODS,
                    {"default": METHOD_CUSTOM},
                ),
                "interpolation": (
                    list(INTERPOLATIONS),
                    {"default": "Lanczos"},
                ),
                "divisor": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1024,
                    "step": 1,
                }),
            },
            "optional": {
                "image_1": ("IMAGE",),
                **{
                    f"image_{index}": ("IMAGE",)
                    for index in range(2, 9)
                },
                "mask_1": ("MASK",),
            },
        }

    RETURN_TYPES = (
        "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "MASK", "MASK", "INT", "INT", "STRING",
        "IMAGE",
    )

    RETURN_NAMES = (
        "图1 / Image 1",
        "图2 / Image 2",
        "图3 / Image 3",
        "图4 / Image 4",
        "图5 / Image 5",
        "图6 / Image 6",
        "图7 / Image 7",
        "图8 / Image 8",
        "图1遮罩 / Mask 1",
        "图1拓展遮罩 / Padding mask 1",
        "图1宽度 / Width 1",
        "图1高度 / Height 1",
        "执行报告 / Report",
        "多图预览 / Combined preview",
    )

    FUNCTION = "resize"
    CATEGORY = "孤海工具箱/图像"

    def resize(
        self,
        language,
        resize_mode,
        megapixels,
        area_multiplier,
        color_preset,
        fill_color,
        ratio_preset,
        aspect_ratio,
        manual_width,
        manual_height,
        edge_length,
        crop_mode,
        resize_method,
        interpolation,
        divisor,
        image_1=None,
        mask_1=None,
        **kwargs,
    ):
        check_interrupt()
        chinese = language != "English"

        inputs = [image_1] + [
            kwargs.get(f"image_{index}")
            for index in range(2, 9)
        ]

        # 图1是后续缩放规则的参考输入；缺失时返回空结果，避免校验时报错。
        if image_1 is None:
            empty_images = torch.empty((0, 1, 1, 3), dtype=torch.float32)
            empty_masks = torch.empty((0, 1, 1), dtype=torch.float32)
            empty_preview = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            report = (
                "没有图1输入，输出为空。"
                if chinese else
                "Image 1 is missing; outputs are empty."
            )
            return (
                empty_images, empty_images, empty_images, empty_images,
                empty_images, empty_images, empty_images, empty_images,
                empty_masks, empty_masks, 0, 0, report, empty_preview,
            )

        if resize_mode not in MODES:
            raise ValueError(
                "未知缩放模式 / Unknown resize mode."
            )
        if color_preset not in (*COLOR_PRESETS, CUSTOM_COLOR):
            raise ValueError(
                "未知颜色预设 / Unknown color preset."
            )
        if interpolation not in INTERPOLATIONS:
            raise ValueError(
                "未知缩放插值 / Unknown interpolation."
            )

        image_interpolation = INTERPOLATIONS[interpolation]
        divisor = int(divisor)

        if divisor < 0:
            raise ValueError(
                "整除数不能为负 / Divisor cannot be negative."
            )

        first_tensor = validate_image(image_1, "图1")
        source_ratio_1 = (
            int(first_tensor.shape[2])
            / int(first_tensor.shape[1])
        )

        limit_area = None
        if resize_mode != MODE_MANUAL:
            megapixels = float(megapixels)
            if not math.isfinite(megapixels) or megapixels <= 0:
                raise ValueError(
                    "百万像素必须是正数 / Megapixels must be positive."
                )

            limit_area = max(
                1,
                int(megapixels * 1_000_000),
            )

            if divisor > 1 and divisor * divisor > limit_area:
                raise ValueError(
                    "整除数平方超过像素上限 / "
                    "Divisor exceeds pixel budget."
                )

        multiplier = float(area_multiplier)
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(
                "其余图像面积乘数必须大于0 / "
                "Other image area multiplier must be positive."
            )

        width = height = edge = 1

        if resize_mode == MODE_MANUAL:
            if resize_method not in METHODS or crop_mode not in FIT_MODES:
                raise ValueError(
                    "未知手动缩放选项 / Unknown manual resize option."
                )

            if resize_method == METHOD_CUSTOM:
                width = int(manual_width)
                height = int(manual_height)
                if width <= 0 or height <= 0:
                    raise ValueError(
                        "手动宽高必须大于0 / "
                        "Width and height must be positive."
                    )
            else:
                edge = int(edge_length)
                if edge <= 0:
                    raise ValueError(
                        "目标边长必须大于0 / "
                        "Target edge must be positive."
                    )

        target_ratio = None
        if resize_mode == MODE_PAD:
            if ratio_preset not in RATIO_PRESETS:
                raise ValueError(
                    "未知比例预设 / Unknown ratio preset."
                )

            if ratio_preset == "原图":
                # 图1的原始比例作为图1及其余图片的统一目标比例。
                target_ratio = source_ratio_1
            elif ratio_preset == "手动":
                target_ratio = parse_ratio(aspect_ratio)
            else:
                target_ratio = parse_ratio(ratio_preset)

        color = (
            parse_color(fill_color)
            if color_preset == CUSTOM_COLOR
            else COLOR_PRESETS[color_preset]
        )

        outputs = [None] * MAX_IMAGES
        reports = []
        first_masks = []
        first_padding = []
        first_width = first_height = 0
        first_area = None

        reports.append(
            f"图像插值：{interpolation}；遮罩插值：双线性。"
            if chinese else
            f"Image interpolation: {interpolation}; mask interpolation: bilinear."
        )

        for slot, value in enumerate(inputs):
            check_interrupt()
            if value is None:
                continue

            tensor = validate_image(value, f"图{slot + 1}")
            batch_size = int(tensor.shape[0])
            masks = (
                prepare_masks(mask_1, batch_size)
                if slot == 0 else None
            )

            original_size = (
                int(tensor.shape[2]),
                int(tensor.shape[1]),
            )
            processed = []

            for index, frame in enumerate(tensor):
                check_interrupt()

                image = tensor_to_pil(frame, color)

                # 对图1逐张判断：未输入遮罩或原始遮罩全为0，
                # 则在最终处理完成后输出纯白遮罩。
                # 在遮罩缩放前检测，不要求其分辨率与图1一致。
                force_white_mask = False

                if slot == 0:
                    if masks is None:
                        force_white_mask = True
                    else:
                        mask_index = (
                            0 if masks.shape[0] == 1 else index
                        )
                        source_mask = masks[mask_index]
                        force_white_mask = (
                            torch.count_nonzero(source_mask).item() == 0
                        )

                mask = (
                    mask_to_pil(masks, index, image.size)
                    if slot == 0 else None
                )

                if resize_mode == MODE_MANUAL and slot == 0:
                    image, mask, padding = manual_resize(
                        image=image,
                        mask=mask,
                        width=width,
                        height=height,
                        edge=edge,
                        method=resize_method,
                        fit=crop_mode,
                        color=color,
                        divisor=divisor,
                        interpolation=image_interpolation,
                    )

                elif resize_mode == MODE_PAD:
                    image, mask, padding = pad_to_ratio(
                        image=image,
                        mask=mask,
                        ratio=target_ratio,
                        color=color,
                        area=limit_area,
                        divisor=divisor,
                        interpolation=image_interpolation,
                    )

                else:
                    if resize_mode == MODE_MANUAL:
                        # 图1处理完成后，其他图片只按图1面积限幅。
                        area = first_area
                        force = False

                    elif resize_mode == MODE_LIMIT and slot > 0:
                        # 其他图片按图1面积乘数和百万像素上限中的较小值缩小。
                        area = min(
                            limit_area,
                            max(1, int(first_area * multiplier)),
                        )
                        force = False

                    else:
                        area = limit_area
                        force = resize_mode == MODE_FORCE

                    image, mask = area_resize(
                        image=image,
                        mask=mask,
                        area=area,
                        divisor=divisor,
                        interpolation=image_interpolation,
                        force=force,
                    )
                    padding = Image.new("L", image.size, 0)

                if slot == 0 and force_white_mask:
                    # 使用最终图像尺寸生成纯白遮罩，
                    # 同时覆盖缩放、裁剪和填充后的结果。
                    mask = Image.new("L", image.size, 255)

                processed.append(image)

                if slot == 0:
                    first_masks.append(mask)
                    first_padding.append(padding)

            output_width, output_height = processed[0].size
            outputs[slot] = images_to_tensor(processed)

            if slot == 0:
                first_width = output_width
                first_height = output_height
                first_area = output_width * output_height

            if chinese:
                reports.append(
                    f"图{slot + 1}：{batch_size}张，"
                    f"{original_size[0]}x{original_size[1]} → "
                    f"{output_width}x{output_height}，"
                    f"{output_width * output_height / 1_000_000:.4f} MP"
                )
            else:
                reports.append(
                    f"Image {slot + 1}: batch {batch_size}, "
                    f"{original_size[0]}x{original_size[1]} -> "
                    f"{output_width}x{output_height}, "
                    f"{output_width * output_height / 1_000_000:.4f} MP"
                )

        if resize_mode == MODE_LIMIT:
            reports.append(
                f"图1按百万像素上限处理；图2～图8仅在面积超过 "
                f"min(上限, 图1最终面积×{multiplier:g}) 时缩小。"
                if chinese else
                f"Image 1 follows the MP limit; images 2-8 shrink only "
                f"above min(limit, image 1 final area × {multiplier:g})."
            )

        elif resize_mode == MODE_FORCE:
            reports.append(
                "所有已连接图片按目标面积缩放，小图允许放大。"
                if chinese else
                "All connected images resize to the target area; upscaling is allowed."
            )

        elif resize_mode == MODE_MANUAL:
            reports.append(
                "手动模式仅控制图1，不使用百万像素上限；"
                "图2～图8只在面积超过图1最终面积时缩小。"
                if chinese else
                "Manual mode controls image 1 without the MP limit; "
                "images 2-8 shrink only when exceeding image 1 final area."
            )

            if resize_method != METHOD_CUSTOM:
                reports.append(
                    f"将边缩放到：{edge}；手动宽高和裁剪模式不参与处理。"
                    if chinese else
                    f"Target edge: {edge}; custom dimensions and fit mode are ignored."
                )

        elif resize_mode == MODE_PAD:
            if ratio_preset == "原图":
                reports.append(
                    "比例设为原图：图1保持自身比例，"
                    "图2～图8填充到图1原始比例。"
                    if chinese else
                    "Original ratio: image 1 keeps its ratio; "
                    "images 2-8 pad to image 1 original ratio."
                )
            else:
                reports.append(
                    f"统一填充比例：{target_ratio:.6f}。"
                    if chinese else
                    f"Shared padding ratio: {target_ratio:.6f}."
                )

        if divisor > 1:
            reports.append(
                "整除数仅在必要时调整尺寸；"
                "未超出百万像素上限但宽高不满足整除要求时，"
                if chinese else
                "The divisor is applied when needed; images below "
                "the megapixel limit are still adjusted downward "
                "if their dimensions are not divisible by the divisor."
            )

        reports.append(
            "图1遮罩逐张处理：未输入或原始遮罩为纯黑时，"
            "输出与最终图1尺寸一致的纯白遮罩；"
            "其他遮罩正常变换；拓展遮罩独立计算。"
            if chinese else
            "Image 1 masks are handled per frame: missing or "
            "all-zero source masks produce white masks matching "
            "the final image size; other masks transform normally. "
            "Padding masks are calculated separately."
        )

        preview = combine_preview(outputs, color)

        return tuple(outputs) + (
            masks_to_tensor(first_masks),
            masks_to_tensor(first_padding),
            first_width,
            first_height,
            "\n".join(reports),
            preview,
        )


NODE_CLASS_MAPPINGS = {
    "GoohaiMultiImageResize": GoohaiMultiImageResize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoohaiMultiImageResize": "多图统一缩放 / Multi-image Resize",
}
