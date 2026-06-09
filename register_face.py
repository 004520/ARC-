"""
人脸录入脚本 — 摄像头拍照录入，支持多张融合

用法：
    python register_face.py                      # 交互式录入
    python register_face.py --input photo.jpg     # 从图片文件录入
    python register_face.py --batch ./photos/     # 从目录批量录入
"""

import os
import argparse
import cv2
import numpy as np
from datetime import datetime
from inference import FaceEngine
from database import FaceDatabase


def register_from_camera(engine, db, user_id, name, gender, phone, department, num_samples=3):
    """打开摄像头，采集多张人脸并融合注册。"""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[错误] 摄像头打开失败")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        print("[错误] Haar 级联文件加载失败")
        cap.release()
        return False

    collected = []
    print(f"\n请正对摄像头，采集 {num_samples} 张人脸（自动采集）...")

    while len(collected) < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))

        display = frame.copy()
        for x, y, w, h in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        status = f"已采集: {len(collected)}/{num_samples}"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Register Face", display)

        if len(faces) > 0:
            # 取最大的人脸
            areas = [w * h for (x, y, w, h) in faces]
            idx = int(np.argmax(areas))
            fx, fy, fw, fh = faces[idx]
            margin = 0.3
            mx, my = int(fw * margin), int(fh * margin)
            crop = frame[
                max(0, fy - my): fy + fh + my,
                max(0, fx - mx): fx + fw + mx,
            ]
            feat = engine.extract_array(crop)
            collected.append(feat)
            print(f"  采集第 {len(collected)} 张成功")

        key = cv2.waitKey(500) & 0xFF
        if key == 27 or key == ord("q"):
            print("用户取消")
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()

    if len(collected) < num_samples:
        print(f"[警告] 仅采集到 {len(collected)} 张，继续注册...")

    # 多张融合注册
    db.register_multi_features(user_id, name, collected, gender=gender, phone=phone, department=department)
    print(f"\n[OK] 注册成功: {name} ({user_id})，融合了 {len(collected)} 张人脸")
    return True


def register_from_image(engine, db, img_path, user_id, name, gender="", phone="", department=""):
    """从单张图片注册人脸。"""
    if not os.path.isfile(img_path):
        print(f"[错误] 图片不存在: {img_path}")
        return False

    img = cv2.imread(img_path)
    if img is None:
        print(f"[错误] 无法读取图片: {img_path}")
        return False

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        print(f"[错误] 未在图片中检测到人脸: {img_path}")
        return False

    # 取最大的人脸
    areas = [w * h for (x, y, w, h) in faces]
    idx = int(np.argmax(areas))
    fx, fy, fw, fh = faces[idx]
    margin = 0.3
    mx, my = int(fw * margin), int(fh * margin)
    crop = img[
        max(0, fy - my): fy + fh + my,
        max(0, fx - mx): fx + fw + mx,
    ]
    feat = engine.extract_array(crop)
    db.register(user_id, name, feat, gender=gender, phone=phone, department=department)
    print(f"[OK] 注册成功: {name} ({user_id})，来源: {img_path}")
    return True


def register_from_dir(engine, db, directory, prefix="img"):
    """从目录批量录入（每个子目录名作为姓名）。"""
    if not os.path.isdir(directory):
        print(f"[错误] 目录不存在: {directory}")
        return

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    # 如果是平铺目录（所有图片在同一层）
    entries = sorted(os.listdir(directory))
    subdirs = [e for e in entries if os.path.isdir(os.path.join(directory, e))]
    images = [e for e in entries if e.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if subdirs:
        # 子目录模式：每个子目录是一个人的照片
        for subdir in subdirs:
            name = subdir
            user_id = f"{prefix}_{name}"
            img_dir = os.path.join(directory, subdir)
            feats = []
            for fname in sorted(os.listdir(img_dir)):
                fpath = os.path.join(img_dir, fname)
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                img = cv2.imread(fpath)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))
                if len(faces) == 0:
                    continue
                areas = [w * h for (x, y, w, h) in faces]
                idx = int(np.argmax(areas))
                fx, fy, fw, fh = faces[idx]
                mx, my = int(fw * 0.3), int(fh * 0.3)
                crop = img[max(0, fy - my): fy + fh + my, max(0, fx - mx): fx + fw + mx]
                feat = engine.extract_array(crop)
                feats.append(feat)

            if feats:
                db.register_multi_features(user_id, name, feats)
                print(f"  [OK] {name}: {len(feats)} 张照片已融合注册")
                count += 1
            else:
                print(f"  [跳过] {name}: 未检测到人脸")

    elif images:
        # 平铺模式：逐一注册
        for i, fname in enumerate(images):
            name = os.path.splitext(fname)[0]
            user_id = f"{prefix}_{name}"
            fpath = os.path.join(directory, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))
            if len(faces) == 0:
                print(f"  [跳过] {fname}: 未检测到人脸")
                continue
            areas = [w * h for (x, y, w, h) in faces]
            idx = int(np.argmax(areas))
            fx, fy, fw, fh = faces[idx]
            mx, my = int(fw * 0.3), int(fh * 0.3)
            crop = img[max(0, fy - my): fy + fh + my, max(0, fx - mx): fx + fw + mx]
            feat = engine.extract_array(crop)
            db.register(user_id, name, feat)
            print(f"  [OK] {name} ({user_id})")
            count += 1
    else:
        print("[警告] 目录中无图片或子目录")

    print(f"\n批量注册完成，共注册 {count} 人")


def list_registered(db):
    """列出已注册用户。"""
    users = db.list_users()
    if not users:
        print("数据库中暂无注册用户")
        return
    print(f"\n已注册 {len(users)} 人：")
    print("-" * 70)
    print(f"{'ID':<20} {'姓名':<10} {'性别':<6} {'电话':<15} {'部门':<15}")
    print("-" * 70)
    for u in users:
        print(f"{u['user_id']:<20} {u['name']:<10} {u['gender']:<6} {u['phone']:<15} {u['department']:<15}")


def delete_user(db, user_id):
    """删除指定用户。"""
    info = db.get_user_info(user_id=user_id)
    if not info:
        print(f"[错误] 用户不存在: {user_id}")
        return
    name = info[0]["name"]
    db.delete_user(user_id)
    print(f"[OK] 已删除用户: {name} ({user_id})")


def interactive_mode(engine, db):
    """交互式命令行界面。"""
    print("\n" + "=" * 50)
    print("  人脸识别系统 - 管理控制台")
    print("=" * 50)

    while True:
        print("\n操作选项：")
        print("  1. 摄像头录入人脸")
        print("  2. 从图片录入人脸")
        print("  3. 批量录入（目录）")
        print("  4. 查看已注册用户")
        print("  5. 删除用户")
        print("  6. 查看识别日志")
        print("  7. 统计信息")
        print("  0. 退出")

        choice = input("\n请输入选项编号: ").strip()

        if choice == "1":
            user_id = input("用户ID (如 zhangsan): ").strip()
            name = input("姓名: ").strip()
            if not user_id or not name:
                print("[错误] 用户ID和姓名不能为空")
                continue
            gender = input("性别 (男/女，回车跳过): ").strip() or "其他"
            phone = input("电话 (回车跳过): ").strip()
            department = input("部门/班级 (回车跳过): ").strip()
            register_from_camera(engine, db, user_id, name, gender, phone, department)

        elif choice == "2":
            img_path = input("图片路径: ").strip().strip('"')
            user_id = input("用户ID: ").strip()
            name = input("姓名: ").strip()
            if not img_path or not user_id or not name:
                print("[错误] 路径、用户ID和姓名不能为空")
                continue
            register_from_image(engine, db, img_path, user_id, name)

        elif choice == "3":
            directory = input("图片目录路径: ").strip().strip('"')
            if not directory:
                print("[错误] 路径不能为空")
                continue
            register_from_dir(engine, db, directory)

        elif choice == "4":
            list_registered(db)

        elif choice == "5":
            list_registered(db)
            user_id = input("输入要删除的用户ID: ").strip()
            if user_id:
                delete_user(db, user_id)

        elif choice == "6":
            logs = db.get_logs(limit=20)
            if not logs:
                print("暂无识别日志")
            else:
                for log in logs:
                    print(f"  [{log['created_at']}] {log['name'] or 'Unknown'} 相似度={log['similarity']:.3f}")

        elif choice == "7":
            print(f"  注册人数: {db.count()}")
            print(f"  识别日志: {db.count_logs()} 条")

        elif choice == "0":
            print("再见！")
            break
        else:
            print("无效选项，请重新输入")


def main():
    parser = argparse.ArgumentParser(description="人脸录入与管理")
    parser.add_argument("--input", help="从单张图片录入")
    parser.add_argument("--batch", help="从目录批量录入")
    parser.add_argument("--list", action="store_true", help="列出已注册用户")
    parser.add_argument("--delete", help="删除指定用户ID")
    parser.add_argument("--user-id", help="指定用户ID（配合 --input 使用）")
    parser.add_argument("--name", help="指定姓名（配合 --input 使用）")
    parser.add_argument("--num-samples", type=int, default=3, help="摄像头采集张数 (默认 3)")
    args = parser.parse_args()

    print("加载模型中...")
    engine = FaceEngine()
    print(f"连接数据库中...")
    db = FaceDatabase()

    if args.list:
        list_registered(db)
    elif args.delete:
        delete_user(db, args.delete)
    elif args.input:
        user_id = args.user_id or input("用户ID: ").strip()
        name = args.name or input("姓名: ").strip()
        register_from_image(engine, db, args.input, user_id, name)
    elif args.batch:
        register_from_dir(engine, db, args.batch)
    else:
        interactive_mode(engine, db)

    db.close()


if __name__ == "__main__":
    main()
