"""
实时摄像头人脸检测与识别

依赖安装：
    pip install opencv-python numpy

可选（更高精度检测）：
    pip install mediapipe          # MediaPipe Face Detection（推荐）
    pip install mtcnn              # MTCNN 检测器（备选）

用法：
    python realtime.py                        # 默认 Haar 级联检测器
    python realtime.py --detector mediapipe  # 使用 MediaPipe（推荐）
    python realtime.py --detector mtcnn      # 使用 MTCNN
    python realtime.py --camera 1            # 指定摄像头编号
    python realtime.py --skip 2              # 每 3 帧检测一次（省算力）
"""

import argparse
import time
import cv2
import numpy as np

from inference import FaceEngine
from database import FaceDatabase


# ── 人脸检测器 ────────────────────────────────────────────

class FaceDetector:
    """人脸检测器统一接口。子类需实现 detect(frame) 方法。"""

    def detect(self, frame):
        """检测人脸。

        Args:
            frame: BGR numpy 数组 (H, W, 3)

        Returns:
            list of (x1, y1, x2, y2, confidence)
        """
        raise NotImplementedError


class HaarDetector(FaceDetector):
    """OpenCV 内置 Haar 级联（零依赖，速度最快，精度一般）。

    适合：快速验证、低配机器、无网络环境。
    """

    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise FileNotFoundError(f"Haar 级联文件未找到: {path}")

    def detect(self, frame, min_size=(80, 80)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=min_size
        )
        return [(x, y, x + w, y + h, 1.0) for x, y, w, h in faces]


class DNNSSDDetector(FaceDetector):
    """OpenCV DNN ResNet-10 SSD 检测器（精度高，速度快）。

    模型文件需提前下载到 models/ 目录：
        https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830
        - res10_300x300_ssd_iter_140000.caffemodel
        - deploy.prototxt
    """

    def __init__(self, model_dir="models"):
        import os
        proto = os.path.join(model_dir, "deploy.prototxt")
        weights = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"模型文件未找到: {weights}\n"
                f"请从 https://github.com/opencv/opencv_3rdparty 下载到 {model_dir}/"
            )
        self.net = cv2.dnn.readNetFromCaffe(proto, weights)
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detect(self, frame, conf_threshold=0.6):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        out = self.net.forward()
        results = []
        for i in range(out.shape[2]):
            conf = float(out[0, 0, i, 2])
            if conf >= conf_threshold:
                x1 = max(0, int(out[0, 0, i, 3] * w))
                y1 = max(0, int(out[0, 0, i, 4] * h))
                x2 = min(w, int(out[0, 0, i, 5] * w))
                y2 = min(h, int(out[0, 0, i, 6] * h))
                results.append((x1, y1, x2, y2, conf))
        return results


class MediaPipeDetector(FaceDetector):
    """Google MediaPipe Face Detection（推荐：精度高，跨平台）。

    pip install mediapipe
    """

    def __init__(self, min_confidence=0.5):
        import mediapipe as mp
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_confidence
        )

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)
        if not result.detections:
            return []
        faces = []
        for det in result.detections:
            bb = det.location_data.relative_bounding_box
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            x2 = min(w, int((bb.xmin + bb.width) * w))
            y2 = min(h, int((bb.ymin + bb.height) * h))
            faces.append((x1, y1, x2, y2, det.score[0]))
        return faces

    def __del__(self):
        if hasattr(self, "detector"):
            self.detector.close()


class MTCNNDetector(FaceDetector):
    """MTCNN 检测器（精度高，速度中等）。

    pip install mtcnn
    """

    def __init__(self, min_face_size=80, thresholds=(0.6, 0.7, 0.7)):
        from mtcnn import MTCNN
        self.mtcnn = MTCNN(min_face_size=min_face_size, thresholds=list(thresholds))

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mtcnn.detect_faces(rgb)
        return [
            (r["box"][0], r["box"][1], r["box"][0] + r["box"][2], r["box"][1] + r["box"][3], r["confidence"])
            for r in results
        ]


# ── 检测器工厂 ──────────────────────────────────────────

DETECTOR_MAP = {
    "haar": HaarDetector,
    "dnn": DNNSSDDetector,
    "mediapipe": MediaPipeDetector,
    "mtcnn": MTCNNDetector,
}


def build_detector(name):
    """根据名称创建检测器实例。"""
    cls = DETECTOR_MAP.get(name)
    if cls is None:
        raise ValueError(f"未知检测器: {name}，可选: {list(DETECTOR_MAP.keys())}")
    return cls()


# ── 实时识别器 ──────────────────────────────────────────

class RealtimeRecognizer:
    """摄像头实时人脸检测与识别。"""

    # 颜色配置 (BGR)
    COLOR_KNOWN    = (0, 255, 0)      # 绿色 — 已识别
    COLOR_UNKNOWN  = (0, 0, 255)       # 红色 — 未知
    COLOR_DETECT   = (255, 200, 0)     # 黄色 — 仅检测模式

    def __init__(
        self,
        engine=None,
        detector="haar",
        camera=0,
        threshold=0.45,
        recognize=True,
        skip_frames=0,
        margin=0.3,
    ):
        self.engine = engine or FaceEngine()
        self.detector = build_detector(detector)
        self.threshold = threshold
        self.recognize = recognize
        self.skip_frames = skip_frames
        self.margin = margin
        self.frame_count = 0

        # 预加载数据库 gallery（向量化搜索）
        if self.recognize:
            self.gallery_features = None
            self.gallery_ids = []
            self.gallery_names = {}
            self._load_gallery()

    def _load_gallery(self):
        """从数据库加载全部特征到内存（向量化搜索只需一次 np.dot）。"""
        try:
            db = FaceDatabase()
            features, ids = db.load_all_features()
            # 加载 name 映射（复用同一连接）
            db.cursor.execute("SELECT user_id, name FROM face_features")
            self.gallery_names = {uid: name for uid, name in db.cursor.fetchall()}
            db.close()
            if len(ids) == 0:
                print("[警告] 数据库为空，只能检测人脸，无法识别身份")
                self.recognize = False
                return
            self.gallery_features = features
            self.gallery_ids = ids
            print(f"Gallery 已加载: {len(ids)} 人")
        except Exception as e:
            print(f"[警告] 数据库连接失败 ({e})，仅检测模式")
            self.recognize = False

    @staticmethod
    def _align_crop(frame, x1, y1, x2, y2, margin=0.3):
        """裁剪人脸区域并适当扩展边距。"""
        h, w = frame.shape[:2]
        fw, fh = x2 - x1, y2 - y1
        mx = int(fw * margin)
        my = int(fh * margin)
        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)
        return frame[cy1:cy2, cx1:cx2]

    def _match(self, face_crop):
        """单张人脸匹配。"""
        feat = self.engine.extract_array(face_crop)
        sims = self.gallery_features @ feat
        idx = int(np.argmax(sims))
        max_sim = float(sims[idx])
        if max_sim >= self.threshold:
            uid = self.gallery_ids[idx]
            return uid, self.gallery_names.get(uid, uid), max_sim
        return None, None, max_sim

    def process_frame(self, frame):
        """处理单帧：检测 + 识别 + 绘制。"""
        # 帧跳过逻辑（省算力）
        if self.skip_frames > 0 and self.frame_count % (self.skip_frames + 1) != 0:
            return frame, []
        self.frame_count += 1

        faces = self.detector.detect(frame)
        results = []

        for x1, y1, x2, y2, conf in faces:
            face_crop = self._align_crop(frame, x1, y1, x2, y2, self.margin)
            if face_crop.size == 0:
                continue

            uid, name, sim = None, None, 0.0
            if self.recognize:
                uid, name, sim = self._match(face_crop)

            results.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "user_id": uid,
                "name": name,
                "similarity": sim,
            })

            # 绘制
            if uid:
                color, label = self.COLOR_KNOWN, f"{name} ({sim:.2f})"
            elif self.recognize:
                color, label = self.COLOR_UNKNOWN, f"Unknown ({sim:.2f})"
            else:
                color, label = self.COLOR_DETECT, f"Face {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # 标签背景
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thickness = 0.6, 1
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

        return frame, results

    def run(self, camera=0, title="Face Recognition"):
        """启动摄像头实时循环。"""
        cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(f"摄像头 {camera} 打开失败")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("按 Q 退出 | 按 R 注册当前帧中的人脸 | 按 N 切换仅检测模式")

        fps_time = time.time()
        fps_val = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, results = self.process_frame(frame)

            # FPS 统计
            now = time.time()
            if now - fps_time >= 1.0:
                fps_val = 1.0 / max(now - fps_time, 1e-6)
                fps_time = now

            # 左上角信息
            mode = "识别模式" if self.recognize else "仅检测"
            info = f"FPS: {fps_val:.1f} | {mode} | 检测到: {len(results)} 人"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("n"):
                self.recognize = not self.recognize
                print(f"切换为: {'识别模式' if self.recognize else '仅检测'}")
            elif key == ord("r"):
                self._register_from_frame(frame, results)

        cap.release()
        cv2.destroyAllWindows()

    def _register_from_frame(self, frame, results):
        """从当前帧检测结果中注册新用户。"""
        if not results:
            print("当前帧无人脸")
            return
        try:
            db = FaceDatabase()
            for i, r in enumerate(results):
                if not r["user_id"]:
                    x1, y1, x2, y2 = r["bbox"]
                    crop = self._align_crop(frame, x1, y1, x2, y2, self.margin)
                    feat = self.engine.extract_array(crop)
                    uid = f"user_{int(time.time())}_{i}"
                    name = input(f"为第 {i+1} 张人脸输入姓名 (回车跳过): ").strip()
                    if name:
                        db.register(uid, name, feat)
                        print(f"已注册: {name} ({uid})")
                        # 更新内存 gallery
                        self.gallery_features = np.vstack([self.gallery_features, feat.reshape(1, -1)])
                        self.gallery_ids.append(uid)
                        self.gallery_names[uid] = name
            db.close()
        except Exception as e:
            print(f"注册失败: {e}")


# ── CLI 入口 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="实时人脸检测与识别")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号 (默认 0)")
    parser.add_argument(
        "--detector", choices=list(DETECTOR_MAP.keys()), default="haar",
        help="人脸检测器: haar(默认) / dnn / mediapipe(推荐) / mtcnn",
    )
    parser.add_argument("--threshold", type=float, default=0.45, help="识别阈值 (默认 0.45)")
    parser.add_argument("--skip", type=int, default=0, help="每 N+1 帧检测一次 (默认 0=每帧)")
    parser.add_argument("--no-recognize", action="store_true", help="仅检测不识别")
    args = parser.parse_args()

    engine = FaceEngine()
    recognizer = RealtimeRecognizer(
        engine=engine,
        detector=args.detector,
        camera=args.camera,
        threshold=args.threshold,
        recognize=not args.no_recognize,
        skip_frames=args.skip,
    )
    recognizer.run(camera=args.camera)


if __name__ == "__main__":
    main()
