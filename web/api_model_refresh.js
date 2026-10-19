import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


function findWidget(node, name) {
    return node.widgets?.find(
        widget => widget.name === name
    );
}


function chooseModel(node, models) {
    return new Promise(resolve => {
        const overlay = document.createElement("div");

        overlay.style.cssText = `
            position: fixed;
            inset: 0;
            z-index: 100000;
            background: rgba(0, 0, 0, 0.55);
            display: flex;
            align-items: center;
            justify-content: center;
        `;

        const panel = document.createElement("div");

        panel.style.cssText = `
            background: #252525;
            color: white;
            padding: 20px;
            border-radius: 8px;
            width: min(520px, 90vw);
            max-height: 85vh;
            overflow: auto;
        `;

        const title = document.createElement("h3");
        title.textContent = `选择模型（${models.length}）`;

        const filter = document.createElement("input");
        filter.placeholder = "输入关键词筛选模型";

        filter.style.cssText = `
            width: 100%;
            box-sizing: border-box;
            margin-bottom: 10px;
            padding: 8px;
            color: white;
            background: #333;
            border: 1px solid #666;
        `;

        const select = document.createElement("select");
        select.size = 14;

        select.style.cssText = `
            width: 100%;
            box-sizing: border-box;
            padding: 6px;
            color: white;
            background: #333;
            border: 1px solid #666;
        `;

        const modelWidget = findWidget(node, "模型名称");
        const currentModel = modelWidget?.value;

        function renderOptions() {
            const keyword = filter.value.trim().toLowerCase();

            select.replaceChildren();

            for (const model of models) {
                if (!model.toLowerCase().includes(keyword)) {
                    continue;
                }

                const option = document.createElement("option");
                option.value = model;
                option.textContent = model;
                select.appendChild(option);
            }

            if (
                Array.from(select.options).some(
                    option => option.value === currentModel
                )
            ) {
                select.value = currentModel;
            } else if (select.options.length) {
                select.selectedIndex = 0;
            }
        }

        filter.addEventListener("input", renderOptions);

        const controls = document.createElement("div");

        controls.style.cssText = `
            display: flex;
            gap: 10px;
            margin-top: 12px;
        `;

        const apply = document.createElement("button");
        apply.textContent = "使用选中的模型";

        const cancel = document.createElement("button");
        cancel.textContent = "取消";

        function close() {
            document.removeEventListener("keydown", onKeyDown);
            overlay.remove();
            resolve();
        }

        function applySelection() {
            if (!select.value || !modelWidget) {
                return;
            }

            modelWidget.value = select.value;

            if (typeof modelWidget.callback === "function") {
                modelWidget.callback(
                    select.value,
                    app.canvas,
                    node
                );
            }

            node.setDirtyCanvas?.(true, true);
            close();
        }

        function onKeyDown(event) {
            if (event.key === "Escape") {
                close();
            }
        }

        apply.addEventListener("click", applySelection);
        cancel.addEventListener("click", close);
        select.addEventListener("dblclick", applySelection);

        overlay.addEventListener("click", event => {
            if (event.target === overlay) {
                close();
            }
        });

        document.addEventListener("keydown", onKeyDown);

        controls.append(apply, cancel);
        panel.append(title, filter, select, controls);
        overlay.appendChild(panel);
        document.body.appendChild(overlay);

        renderOptions();
        filter.focus();
    });
}


app.registerExtension({
    name: "MiniAPILLM.ModelRefresh",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MiniAPILLMLoader") {
            return;
        }

        const original = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments);
            const node = this;

            let busy = false;

            const button = node.addWidget(
                "button",
                "刷新模型列表",
                null,
                async () => {
                    if (busy) {
                        return;
                    }

                    const baseWidget = findWidget(node, "接口地址");
                    const keyWidget = findWidget(node, "接口密钥");

                    // 这里只读取加载器自身的控件，
                    // 不全局扫描 DOM，避免拿到其他节点的密钥。
                    const baseUrl = String(
                        baseWidget?.value ?? ""
                    ).trim();

                    const apiKey = String(
                        keyWidget?.value ?? ""
                    ).trim();

                    if (!baseUrl) {
                        alert("请先填写 Base URL。");
                        return;
                    }

                    busy = true;
                    button.name = "正在刷新...";
                    node.setDirtyCanvas?.(true, true);

                    try {
                        const response = await api.fetchApi(
                            "/mini_api_llm/models",
                            {
                                method: "POST",
                                headers: {
                                    "Content-Type": "application/json",
                                },
                                body: JSON.stringify({
                                    base_url: baseUrl,
                                    api_key: apiKey,
                                }),
                            }
                        );

                        let data;

                        try {
                            data = await response.json();
                        } catch {
                            throw new Error(
                                "模型列表接口没有返回有效 JSON。"
                            );
                        }

                        if (!response.ok || data.error) {
                            throw new Error(
                                data.error
                                || `HTTP ${response.status}`
                            );
                        }

                        if (
                            !Array.isArray(data.models)
                            || !data.models.length
                        ) {
                            throw new Error("没有获取到模型。");
                        }

                        await chooseModel(node, data.models);

                    } catch (error) {
                        alert(`刷新模型失败：${error.message}`);

                    } finally {
                        busy = false;
                        button.name = "刷新模型列表";
                        node.setDirtyCanvas?.(true, true);
                    }
                },
                {
                    serialize: false,
                }
            );

            node.setSize?.(node.computeSize());

            return result;
        };
    },
});
