import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
from ArcFace.model import FaceEncoder

# 项目根目录（inference.py 所在目录）
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "checkpoints" / "best_face_model.pth"


def export_torchscript(
    model_path=str(DEFAULT_MODEL),
    output_path=None,
    device=None,
):
    """将 .pth 模型转为 TorchScript (.pt)，推理速度提升 30-50%。

    优势：去除 Python 动态开销，融合算子，支持 C++ 部署。
    用法：python inference.py --export
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if output_path is None:
        output_path = str(DEFAULT_MODEL).replace(".pth", ".pt")

    model = FaceEncoder(embedding_dim=512, use_se=True, use_sa=True).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False)
    )
    model.eval()

    dummy = torch.randn(1, 3, 112, 112, device=device)
    scripted = torch.jit.trace(model, dummy)
    scripted.save(output_path)
    print(f"TorchScript 已保存: {output_path}  ({os.path.getsize(output_path)/1024/1024:.1f} MB)")


class FaceEngine:
    """统一人脸推理引擎，整合特征提取、1:1 验证、1:N 识别。

    模型加载优先级：
      1. TorchScript (.pt) — 更快，推荐生产使用
      2. PyTorch (.pth)    — 原始权重，兼容开发调试
    """

    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            model_path = str(DEFAULT_MODEL)

        # 自动探测模型格式：优先 .pt (TorchScript)，回退 .pth
        if not os.path.isfile(model_path):
            alt = model_path.rsplit(".", 1)[0] + (".pt" if model_path.endswith(".pth") else ".pth")
            if os.path.isfile(alt):
                model_path = alt

        self._load_model(model_path)

        # 与训练验证预处理完全对齐
        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_model(self, path):
        """根据文件扩展名自动选择加载方式。"""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pt":
            self.model = torch.jit.load(path, map_location=self.device)
            self.model.eval()
            self._format = "torchscript"
        else:
            self.model = FaceEncoder(embedding_dim=512, use_se=True, use_sa=True).to(self.device)
            self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
            self.model.eval()
            self._format = "pytorch"
        print(f"模型已加载 [{self._format}]: {os.path.basename(path)}  →  设备: {self.device}")

    # ── 特征提取 ────────────────────────────────────────

    def _preprocess_path(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    def _preprocess_array(self, img_array):
        """从 BGR numpy 数组 (H,W,3) 或 RGB numpy 数组预处理。"""
        if img_array.ndim == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
        elif img_array.shape[2] == 3:
            b_mean = float(img_array[..., 0].mean())
            r_mean = float(img_array[..., 2].mean())
            if b_mean > r_mean + 5:
                img_array = img_array[:, :, ::-1]  # BGR → RGB
        img = Image.fromarray(img_array.astype(np.uint8), "RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract(self, img_path):
        """从文件路径提取特征向量 → shape (512,)"""
        tensor = self._preprocess_path(img_path)
        return self.model(tensor).cpu().numpy().flatten()

    @torch.no_grad()
    def extract_array(self, face_crop):
        """从 numpy 数组提取特征向量 → shape (512,)。

        face_crop: numpy 数组 (H, W, 3)，BGR 或 RGB 均可。
        适用场景：摄像头截取的人脸区域直接传入，无需写临时文件。
        """
        tensor = self._preprocess_array(face_crop)
        return self.model(tensor).cpu().numpy().flatten()

    @torch.no_grad()
    def extract_batch(self, img_paths):
        """从文件路径批量提取 → shape (N, 512)"""
        tensors = torch.stack([self._preprocess_path(p) for p in img_paths])
        return self.model(tensors).cpu().numpy()

    @torch.no_grad()
    def extract_batch_array(self, face_crops):
        """从 numpy 数组批量提取 → shape (N, 512)"""
        tensors = torch.stack([self._preprocess_array(c) for c in face_crops])
        return self.model(tensors).cpu().numpy()

    # ── 相似度计算 ──────────────────────────────────────

    @staticmethod
    def similarity(a, b):
        return float(np.dot(a, b))

    @staticmethod
    def similarity_matrix(query, gallery):
        return query @ gallery.T

    # ── 1:1 验证 / 1:N 识别 ────────────────────────────

    def verify(self, img1_path, img2_path, threshold=0.45):
        feat1 = self.extract(img1_path)
        feat2 = self.extract(img2_path)
        sim = self.similarity(feat1, feat2)
        return {"similarity": round(sim, 6), "is_same": sim > threshold}

    def identify(self, query_img, gallery, threshold=0.45):
        """1:N 内存识别（gallery: list of (id, name, feature_vec)）。"""
        query_feat = self.extract(query_img)
        best_match, max_sim = None, -1.0
        for uid, name, feat in gallery:
            sim = float(np.dot(query_feat, feat))
            if sim > max_sim:
                max_sim, best_match = sim, (uid, name)
        if max_sim > threshold:
            return {"user_id": best_match[0], "name": best_match[1],
                    "similarity": round(max_sim, 6), "match": True}
        return None

    def identify_batch(self, query_img, gallery_features, gallery_ids, threshold=0.45):
        """1:N 向量化识别（gallery_features: numpy array, shape (N, 512)）。"""
        query_feat = self.extract(query_img)
        sims = gallery_features @ query_feat
        idx = int(np.argmax(sims))
        max_sim = float(sims[idx])
        if max_sim > threshold:
            return {"user_id": gallery_ids[idx], "similarity": round(max_sim, 6), "match": True}
        return None


if __name__ == "__main__":
    import sys

    if "--export" in sys.argv:
        export_torchscript()
    else:
        engine = FaceEngine()
        print(f"格式: {engine._format} | 设备: {engine.device}")
