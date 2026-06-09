"""
使用训练集图片测试模型性能
测试场景：
1. 同一人的不同变体图片（正脸 vs 侧脸）
2. 不同人的图片对比
"""
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import random
from ArcFace.model import FaceEncoder


class DatasetTester:
    def __init__(self, model_path='checkpoints/best_face_model.pth'):
        """初始化测试器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"️ 使用设备: {self.device}")
        
        # 加载模型（必须与训练时的参数一致）
        self.model = FaceEncoder(embedding_dim=512, use_se=True, use_sa=True).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"✅ 模型加载成功: {model_path}")

        # 图像预处理（与验证集一致）
        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def extract_feature(self, img_path):
        """提取图片特征向量"""
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(tensor)
        return F.normalize(feature).cpu().numpy()

    def compare_images(self, img1_path, img2_path):
        """比对两张图片"""
        feat1 = self.extract_feature(img1_path)
        feat2 = self.extract_feature(img2_path)
        
        cosine_sim = float(np.dot(feat1[0], feat2[0]))
        euclidean_dist = float(np.linalg.norm(feat1[0] - feat2[0]))
        
        return cosine_sim, euclidean_dist

    def test_same_person(self, person_id='000'):
        """测试同一人的不同变体"""
        # 确保 ID 是三位数
        person_id = f"{int(person_id):03d}"
        person_dir = f'D:/face_dataset/Cleaned_CASIA_FaceV5/{person_id}'
        
        # 获取所有图片
        images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png'))]
        
        if len(images) < 2:
            print(f"️ {person_id} 目录下图片不足")
            return
        
        # 随机选择两张不同的图片
        img1, img2 = random.sample(images, 2)
        img1_path = os.path.join(person_dir, img1)
        img2_path = os.path.join(person_dir, img2)
        
        print(f"\n{'='*60}")
        print(f"📋 测试：同一人不同变体 (ID: {person_id})")
        print(f"{'='*60}")
        print(f"   图片1: {img1}")
        print(f"   图片2: {img2}")
        
        cosine_sim, euclidean_dist = self.compare_images(img1_path, img2_path)
        
        print(f"\n 结果:")
        print(f"   余弦相似度: {cosine_sim:.4f}")
        print(f"   欧氏距离: {euclidean_dist:.4f}")
        
        # 判断
        is_same = cosine_sim > 0.35
        print(f"   判断: {'✅ 同一人' if is_same else '❌ 不同人'} (阈值: 0.35)")
        
        return cosine_sim

    def test_different_persons(self, person1_id='000', person2_id='001'):
        """测试不同人的图片"""
        # 确保 ID 是三位数
        person1_id = f"{int(person1_id):03d}"
        person2_id = f"{int(person2_id):03d}"
        person1_dir = f'D:/face_dataset/Cleaned_CASIA_FaceV5/{person1_id}'
        person2_dir = f'D:/face_dataset/Cleaned_CASIA_FaceV5/{person2_id}'
        
        # 获取图片
        img1 = random.choice([f for f in os.listdir(person1_dir) if f.endswith(('.jpg', '.png'))])
        img2 = random.choice([f for f in os.listdir(person2_dir) if f.endswith(('.jpg', '.png'))])
        
        img1_path = os.path.join(person1_dir, img1)
        img2_path = os.path.join(person2_dir, img2)
        
        print(f"\n{'='*60}")
        print(f" 测试：不同人对比 (ID: {person1_id} vs {person2_id})")
        print(f"{'='*60}")
        print(f"   图片1: {person1_id}/{img1}")
        print(f"   图片2: {person2_id}/{img2}")
        
        cosine_sim, euclidean_dist = self.compare_images(img1_path, img2_path)
        
        print(f"\n📊 结果:")
        print(f"   余弦相似度: {cosine_sim:.4f}")
        print(f"   欧氏距离: {euclidean_dist:.4f}")
        
        # 判断
        is_same = cosine_sim > 0.35
        print(f"   判断: {'✅ 同一人' if is_same else '❌ 不同人'} (阈值: 0.35)")
        
        return cosine_sim

    def batch_test(self, num_tests=10):
        """批量测试"""
        print(f"\n{'='*60}")
        print(f" 批量测试 ({num_tests} 组)")
        print(f"{'='*60}")
        
        same_person_scores = []
        different_person_scores = []
        
        # 测试同一人
        for _ in range(num_tests):
            person_id = f"{random.randint(0, 499):03d}"
            score = self.test_same_person(person_id)
            same_person_scores.append(score)
        
        # 测试不同人
        for _ in range(num_tests):
            p1 = f"{random.randint(0, 499):03d}"
            p2 = f"{random.randint(0, 499):03d}"
            while p1 == p2:
                p2 = f"{random.randint(0, 499):03d}"
            score = self.test_different_persons(p1, p2)
            different_person_scores.append(score)
        
        # 统计结果
        print(f"\n{'='*60}")
        print(f"📈 批量测试结果统计")
        print(f"{'='*60}")
        
        same_array = np.array(same_person_scores)
        diff_array = np.array(different_person_scores)
        
        print(f"\n同一人测试 ({num_tests} 组):")
        print(f"   平均余弦相似度: {same_array.mean():.4f}")
        print(f"   标准差: {same_array.std():.4f}")
        print(f"   最高: {same_array.max():.4f}")
        print(f"   最低: {same_array.min():.4f}")
        print(f"   中位数: {np.median(same_array):.4f}")
        correct_rate = (same_array > 0.35).sum() / num_tests * 100
        print(f"   正确识别率 (>0.35): {correct_rate:.1f}%")
        
        print(f"\n不同人测试 ({num_tests} 组):")
        print(f"   平均余弦相似度: {diff_array.mean():.4f}")
        print(f"   标准差: {diff_array.std():.4f}")
        print(f"   最高: {diff_array.max():.4f}")
        print(f"   最低: {diff_array.min():.4f}")
        print(f"   中位数: {np.median(diff_array):.4f}")
        false_positive_rate = (diff_array > 0.35).sum() / num_tests * 100
        print(f"   误判率 (>0.35): {false_positive_rate:.1f}%")
        
        # 综合评估
        separation = same_array.mean() - diff_array.mean()
        print(f"\n{'='*60}")
        print(f"🎯 综合评估")
        print(f"{'='*60}")
        print(f"   类内-类间分离度: {separation:.4f}")
        print(f"   评估: {'✅ 优秀' if separation > 0.3 else '⚠️ 良好' if separation > 0.2 else '❌ 需改进'}")
        print(f"\n   建议阈值范围: {diff_array.mean() + same_array.mean():.4f} ± 0.05")


def main():
    """主函数"""
    tester = DatasetTester()
    
    print("\n" + "="*60)
    print(" 训练集测试系统")
    print("="*60)
    
    # 选择测试模式
    print("\n请选择测试模式:")
    print("1. 测试同一人的不同变体")
    print("2. 测试不同人对比")
    print("3. 批量测试（推荐）")
    print("4. 自定义测试")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    if choice == '1':
        person_id = input("请输入人员ID (0-499): ").strip()
        tester.test_same_person(person_id)
    elif choice == '2':
        p1 = input("请输入第一个人 ID (0-499): ").strip()
        p2 = input("请输入第二个人 ID (0-499): ").strip()
        tester.test_different_persons(p1, p2)
    elif choice == '3':
        num = int(input("请输入测试组数 (建议 10-50): ").strip())
        tester.batch_test(num)
    elif choice == '4':
        img1 = input("请输入第一张图片路径: ").strip()
        img2 = input("请输入第二张图片路径: ").strip()
        cosine_sim, euclidean_dist = tester.compare_images(img1, img2)
        print(f"\n余弦相似度: {cosine_sim:.4f}")
        print(f"欧氏距离: {euclidean_dist:.4f}")
    else:
        print("❌ 无效选项")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 测试被中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
