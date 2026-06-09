"""
人脸识别系统 — 整合测试入口

运行方式：
    python test_simple.py

功能：
    - 实时摄像头画面，框选人脸
    - 菜单 1：录入人脸（输入名字 → 自动采集 → 注册到数据库）
    - 菜单 2：人脸识别（实时匹配，控制台打印名字）
    - ESC 退出程序
"""

import time
import cv2
import numpy as np
import threading
from inference import FaceEngine
from database import FaceDatabase


class FaceApp:
    def __init__(self):
        print("正在加载模型...")
        self.engine = FaceEngine()

        print("正在连接数据库...")
        self.db = FaceDatabase()

        self.threshold = 0.45

        # Haar 级联检测器
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # 摄像头
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("摄像头打开失败，请检查设备")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 共享状态
        self.frame = None
        self.faces = []               # [(x1, y1, x2, y2, conf), ...]
        self.running = True
        self.return_to_menu = False    # Q 键返回主菜单的信号

        # 后台采集线程
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        # 等待第一帧就绪
        while self.frame is None:
            time.sleep(0.05)

    # ── 后台摄像头采集线程 ─────────────────────────────

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self.frame = frame

            # 检测人脸
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            detected = self.cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80)
            )
            faces = []
            for x, y, w, h in detected:
                faces.append((x, y, x + w, y + h, 1.0))
            self.faces = faces

            # 绘制并显示
            display = frame.copy()
            for x1, y1, x2, y2, conf in faces:
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"Face {conf:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("face", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:          # ESC — 退出程序
                self.running = False
            elif key == ord('q'):    # Q — 返回主菜单
                self.return_to_menu = True

        self.cap.release()
        cv2.destroyAllWindows()

    # ── 工具方法 ───────────────────────────────────────

    def _get_best_face(self):
        """返回最大人脸的裁剪区域，无人脸返回 None。"""
        if not self.faces or self.frame is None:
            return None
        best = max(self.faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
        x1, y1, x2, y2, conf = best
        h, w = self.frame.shape[:2]
        margin = 0.3
        mx = int((x2 - x1) * margin)
        my = int((y2 - y1) * margin)
        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)
        crop = self.frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        return crop

    def _wait_for_face(self, timeout=10):
        """等待检测到人脸，返回裁剪区域或 None。"""
        print("请面对摄像头，等待检测人脸...")
        start = time.time()
        while time.time() - start < timeout:
            if not self.running:
                return None
            crop = self._get_best_face()
            if crop is not None:
                return crop
            time.sleep(0.1)
        return None

    # ── 录入人脸 ───────────────────────────────────────

    def register_face(self):
        """录入人脸到数据库。"""
        name = input("请输入姓名: ").strip()
        if not name:
            print("[取消] 姓名不能为空")
            return

        user_id = f"user_{int(time.time())}"

        # 采集多张特征取平均（更稳定）
        features = []
        num_samples = 3
        print(f"请保持正脸，将采集 {num_samples} 张...")

        for i in range(num_samples):
            # 每张间隔 1 秒，确保拿到不同帧
            if i > 0:
                time.sleep(1)
            crop = self._wait_for_face(timeout=8)
            if crop is None:
                print(f"[警告] 第 {i + 1} 张未检测到人脸，跳过")
                continue
            feat = self.engine.extract_array(crop)
            features.append(feat)
            print(f"  采集第 {i + 1}/{num_samples} 张成功")

        if not features:
            print("[失败] 未能采集到任何人脸，请重试")
            return

        # 多张融合注册
        avg_feat = np.mean(features, axis=0)
        self.db.register(user_id, name, avg_feat)

        total = self.db.count()
        print(f"[成功] 已录入: {name} (ID: {user_id})，当前数据库共 {total} 人")

    # ── 人脸识别 ───────────────────────────────────────

    def recognize_face(self):
        """识别一次，打印结果后返回主菜单。"""
        print("正在识别中，请面对摄像头...")

        gallery_features, gallery_ids, gallery_names = self.db.load_all_features_with_names()
        if len(gallery_ids) == 0:
            print("[警告] 数据库为空，请先录入人脸（菜单 1）")
            return

        # 等待检测到人脸
        crop = self._wait_for_face(timeout=10)
        if crop is None:
            print("[未检测到人脸]")
            return

        feat = self.engine.extract_array(crop)
        sims = gallery_features @ feat
        idx = int(np.argmax(sims))
        max_sim = float(sims[idx])

        if max_sim >= self.threshold:
            print(f"识别结果: {gallery_names[idx]} (相似度: {max_sim:.3f})")
        else:
            print("[识别失败] 未匹配到已知人脸")

    # ── 主循环 ─────────────────────────────────────────

    def run(self):
        print("\n" + "=" * 40)
        print("  人脸识别系统")
        print("=" * 40)

        while self.running:
            print("\n" + "-" * 40)
            print("  1. 录入人脸")
            print("  2. 人脸识别")
            print("  ESC 退出程序")
            print("-" * 40)
            choice = input("请选择 (1/2): ").strip()

            if choice == "1":
                self.register_face()
            elif choice == "2":
                self.recognize_face()
            elif choice == "":
                continue
            else:
                print("无效选项，请输入 1 或 2")

        print("程序已退出")


if __name__ == "__main__":
    try:
        app = FaceApp()
        app.run()
    except RuntimeError as e:
        print(f"[错误] {e}")
    except KeyboardInterrupt:
        print("\n程序已中断退出")
