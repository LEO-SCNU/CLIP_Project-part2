import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from torchvision.datasets import CIFAR10
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
import copy

# 1. 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-large-patch14"


def load_model():
    print(f"正在加载模型: {model_id}...")
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    # 保存原始状态以便重置
    original_state = copy.deepcopy(model.state_dict())
    return model, processor, original_state


def evaluate_model(model, processor, description="Evaluating"):
    dataset = CIFAR10(root='./data', train=False, download=True)
    target_map = {3: 0, 5: 1, 2: 2}  # cat:0, dog:1, bird:2
    classes_of_interest = [3, 5, 2]
    filtered_data = [(img, target_map[label]) for img, label in dataset if label in classes_of_interest]

    text_prompts = ['a photo of a cat', 'a photo of a dog', 'a photo of a bird']

    # 提取文本特征
    with torch.no_grad():
        inputs_text = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
        outputs_text = model.get_text_features(**inputs_text)
        text_features = outputs_text if isinstance(outputs_text, torch.Tensor) else outputs_text.pooler_output
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    all_preds = []
    all_labels = []

    for img, label in tqdm(filtered_data, desc=description):
        with torch.no_grad():
            # 双重增强策略
            img_sharp = img.filter(ImageFilter.SHARPEN)
            imgs = [img, img_sharp]

            inputs = processor(images=imgs, return_tensors="pt").to(device)
            outputs_image = model.get_image_features(**inputs)
            image_features = outputs_image if isinstance(outputs_image, torch.Tensor) else outputs_image.pooler_output
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits = (image_features @ text_features.t()) * logit_scale
            probs = torch.softmax(logits, dim=1).mean(dim=0)

            pred = torch.argmax(probs).item()
            all_preds.append(pred)
            all_labels.append(label)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    class_names = ['cat', 'dog', 'bird']

    accs = {}
    for i, name in enumerate(class_names):
        idx = (all_labels == i)
        acc = (all_preds[idx] == all_labels[idx]).mean() if idx.sum() > 0 else 0.0
        accs[name] = acc

    overall_acc = (all_preds == all_labels).mean()
    return accs, overall_acc


# ==================== 基于梯度的精准手术 ====================
def perform_surgery_gradient_based(model, processor, target_class_idx=0, prune_ratio=0.05, intensity=1.0):
    """
    利用梯度定位关键神经元，实施定向失忆
    """
    print(f"\n>>> 正在执行精准手术 (梯度定位法): 目标索引={target_class_idx}")

    # 1. 准备一个代表性的“猫”的样本用于梯度计算
    # 我们使用CIFAR10训练集中的猫来定位
    print(">>> 正在定位关键神经元 (计算梯度中)...")
    dataset = CIFAR10(root='./data', train=True, download=True)
    cat_imgs = [img for img, label in dataset if label == 3][:10]  # 取10张猫图求平均梯度

    # 目标文本特征
    text_prompts = ['a photo of a cat', 'a photo of a dog', 'a photo of a bird']
    inputs_text = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

    # 目标层：Vision Transformer 的最后一层 Transformer Block
    # CLIP ViT-L 的结构：model.vision_model.encoder.layers[23] (最后一层)
    target_layer = model.vision_model.encoder.layers[-1]

    # 存储梯度的容器
    gradients = []

    def hook_fn(module, grad_input, grad_output):
        gradients.append(grad_output[0].clone())

    # 注册Hook
    handle = target_layer.register_backward_hook(hook_fn)

    model.zero_grad()

    # 前向传播 + 反向传播
    inputs = processor(images=cat_imgs, return_tensors="pt").to(device)
    with torch.set_grad_enabled(True):
        # 获取图像特征
        outputs = model.vision_model(pixel_values=inputs['pixel_values'])
        image_embeds = outputs.pooler_output
        image_embeds = model.visual_projection(image_embeds)

        # 计算Logits
        text_feats_out = model.get_text_features(**inputs_text)
        # 兼容性处理：判断返回的是对象还是张量
        if isinstance(text_feats_out, torch.Tensor):
            text_feats = text_feats_out
        else:
            text_feats = text_feats_out.pooler_output

        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits = (image_embeds @ text_feats.t()) * logit_scale

        target_logits = logits[:, target_class_idx]
        loss = -target_logits.mean()  # 最大化降低Cat的得分

        loss.backward()

    handle.remove()

    # 2. 分析梯度，定位关键通道
    # gradients[0] 形状: [Batch, SeqLen, HiddenDim]
    # 我们关注 CLS token (序列第0个) 的梯度
    if not gradients:
        print("错误：未捕获到梯度。")
        return model

    cls_grad = gradients[0][:, 0, :].abs().mean(dim=0)  # 对Batch求平均，取绝对值
    # cls_grad 形状: [HiddenDim] (1024 for ViT-L)

    # 找到梯度最大的前 k 个维度
    k = int(cls_grad.shape[0] * prune_ratio)
    top_k_vals, top_k_indices = torch.topk(cls_grad, k)

    print(f">>> 已定位到前 {k} 个关键神经元，正在执行抑制...")

    # 3. 实施手术
    with torch.no_grad():
        # 方法：修改最后一层的 LayerNorm
        # 如果我们降低 LayerNorm 的权重，该通道的输出会被抑制
        ln = target_layer.layer_norm2

        # 创建一个Mask
        mask = torch.ones_like(ln.weight.data)
        mask[top_k_indices] = (1 - intensity)

        # 应用手术
        ln.weight.data *= mask

    print(">>> 手术完成！")
    return model

# ====================================================================

if __name__ == "__main__":
    model, processor, original_state = load_model()

    # ================= 实验1: Baseline =================
    print("\n" + "#" * 20 + " 阶段一: 基线测试 " + "#" * 20)
    base_accs, base_overall = evaluate_model(model, processor, description="Baseline")
    print(f"{'Class':<12} | {'Accuracy':<10}")
    print("-" * 30)
    for name, acc in base_accs.items():
        print(f"{name:<12} | {acc:.2%}")
    print(f"{'Overall':<12} | {base_overall:.2%}")

    # ================= 实验2: 调参后的遗忘（目标控制cat到60%） =================
    print("\n" + "#" * 20 + " 阶段二: 定向失忆手术 " + "#" * 20)

    # 恢复原始权重 (防止多次运行叠加)
    model.load_state_dict(original_state)
    # 对intensity进行调参
    perform_surgery_gradient_based(model, processor, target_class_idx=0, prune_ratio=0.2, intensity=4.592)

    surg_accs, surg_overall = evaluate_model(model, processor, description="After Surgery")

    print(f"{'Class':<12} | {'Accuracy':<10}")
    print("-" * 30)
    for name, acc in surg_accs.items():
        print(f"{name:<12} | {acc:.2%}")

    print(f"{'Overall':<12} | {surg_overall:.2%}")
