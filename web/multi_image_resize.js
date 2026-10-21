import { app } from "../../scripts/app.js";

const NODE_NAME = "GoohaiMultiImageResize";

const LABELS = {
    language: ["界面语言", "Language"],
    resize_mode: ["缩放模式", "Resize mode"],
    megapixels: ["百万像素上限", "Megapixel limit"],
    area_multiplier: [
        "其余图像根据图1面积乘数",
        "Other images relative to image 1 area",
    ],
    color_preset: ["填充颜色", "Fill color"],
    fill_color: ["自定义RGB颜色", "Custom RGB color"],
    ratio_preset: ["比例预设", "Aspect ratio preset"],
    aspect_ratio: ["手动比例", "Manual aspect ratio"],
    manual_width: ["宽度", "Width"],
    manual_height: ["高度", "Height"],
    edge_length: ["将边缩放到", "Target edge length"],
    crop_mode: ["裁剪模式", "Fit mode"],
    resize_method: ["缩放方式", "Resize method"],
    interpolation: ["缩放插值", "Resize interpolation"],
    divisor: ["整除数", "Divisor"],
    mask_1: ["图1遮罩", "Image 1 mask"],
};

const OUTPUT_LABELS = [
    ["图1", "Image 1"],
    ["图2", "Image 2"],
    ["图3", "Image 3"],
    ["图4", "Image 4"],
    ["图5", "Image 5"],
    ["图6", "Image 6"],
    ["图7", "Image 7"],
    ["图8", "Image 8"],
    ["图1遮罩", "Image 1 mask"],
    ["图1拓展遮罩", "Image 1 padding mask"],
    ["图1宽度", "Image 1 width"],
    ["图1高度", "Image 1 height"],
    ["执行报告", "Report"],
];

const widgetStates = new WeakMap();
const hookedWidgets = new WeakSet();

function findWidget(node, name) {
    return node.widgets?.find(widget => widget.name === name);
}

function valueOf(node, name) {
    return String(findWidget(node, name)?.value ?? "");
}

function setVisible(widget, visible) {
    if (!widget) return;

    // 不干预已经转换为输入端口的控件。
    if (String(widget.type).startsWith("converted-widget")) return;

    if (!widgetStates.has(widget)) {
        widgetStates.set(widget, {
            type: widget.type,
            computeSize: widget.computeSize,
        });
    }

    const original = widgetStates.get(widget);

    if (visible) {
        widget.type = original.type;
        if (original.computeSize === undefined) {
            delete widget.computeSize;
        } else {
            widget.computeSize = original.computeSize;
        }
    } else {
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
    }
}

function updateNode(node) {
    const english = valueOf(node, "language") === "English";
    const languageColumn = english ? 1 : 0;

    for (const widget of node.widgets ?? []) {
        if (LABELS[widget.name]) {
            widget.label = LABELS[widget.name][languageColumn];
        }
    }

    for (const input of node.inputs ?? []) {
        const match = /^image_(\d+)$/.exec(input.name);

        if (match) {
            input.label = english
                ? `Image ${match[1]}`
                : `图${match[1]}`;
        } else if (LABELS[input.name]) {
            input.label = LABELS[input.name][languageColumn];
        }
    }

    for (let index = 0; index < (node.outputs?.length ?? 0); index++) {
        if (OUTPUT_LABELS[index]) {
            node.outputs[index].label = OUTPUT_LABELS[index][languageColumn];
        }
    }

    const mode = valueOf(node, "resize_mode");
    const limit = mode.startsWith("统一百万像素限制");
    const force = mode.startsWith("所有图像强制缩放");
    const manual = mode.startsWith("图1手动尺寸/边模式");
    const padding = mode.startsWith("统一比例填充");

    const resizeMethod = valueOf(node, "resize_method");
    const customSize = resizeMethod.startsWith("自定义宽高");
    const cropMode = valueOf(node, "crop_mode");
    const fitPadding = cropMode.startsWith("填充");

    const needsFillColor = padding || (manual && customSize && fitPadding);
    const isCustomColor = valueOf(node, "color_preset").startsWith("自定义");
    const manualRatio = valueOf(node, "ratio_preset") === "手动";

    const visibility = {
        megapixels: !manual,
        area_multiplier: limit,
        color_preset: needsFillColor,
        fill_color: needsFillColor && isCustomColor,
        ratio_preset: padding,
        aspect_ratio: padding && manualRatio,
        manual_width: manual && customSize,
        manual_height: manual && customSize,
        edge_length: manual && !customSize,
        crop_mode: manual && customSize,
        resize_method: manual,
        interpolation: true,
        divisor: true,
    };

    // 处理 unused 变量，同时保留模式判定易读性。
    void force;

    for (const [name, visible] of Object.entries(visibility)) {
        setVisible(findWidget(node, name), visible);
    }

    if (typeof node.computeSize === "function") {
        const computed = node.computeSize();
        node.setSize([
            Math.max(node.size?.[0] ?? 0, computed[0]),
            computed[1],
        ]);
    }

    node.setDirtyCanvas(true, true);
}

function hookWidget(node, name) {
    const widget = findWidget(node, name);
    if (!widget || hookedWidgets.has(widget)) return;

    hookedWidgets.add(widget);
    const previous = widget.callback;

    widget.callback = function (...args) {
        try {
            return previous?.apply(this, args);
        } finally {
            updateNode(node);
        }
    };
}

function initialize(node) {
    for (const name of [
        "language",
        "resize_mode",
        "resize_method",
        "crop_mode",
        "ratio_preset",
        "color_preset",
    ]) {
        hookWidget(node, name);
    }

    updateNode(node);
}

app.registerExtension({
    name: "Goohai.MultiImageResize.UI",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const previousCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = previousCreated?.apply(this, args);
            queueMicrotask(() => initialize(this));
            return result;
        };

        const previousConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = previousConfigure?.apply(this, args);
            queueMicrotask(() => initialize(this));
            return result;
        };
    },
});
