"""
训练监控脚本 - 实时显示训练进度和指标
"""
import os
import time
from datetime import datetime

def monitor_training():
    """监控训练过程"""
    print("="*60)
    print("📊 训练监控系统")
    print("="*60)
    
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        print("❌ checkpoints 目录不存在，请先开始训练")
        return
    
    print(f"\n⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 监控目录: {os.path.abspath(checkpoint_dir)}")
    print("\n💡 提示:")
    print("   - 此脚本会每10秒检查一次训练状态")
    print("   - 按 Ctrl+C 停止监控")
    print("   - 训练完成后会自动检测最佳模型\n")
    
    last_size = 0
    start_time = time.time()
    
    try:
        while True:
            # 检查最新检查点
            latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            best_model = os.path.join(checkpoint_dir, 'best_face_model.pth')
            
            current_time = time.time()
            elapsed = current_time - start_time
            
            if os.path.exists(latest_checkpoint):
                size = os.path.getsize(latest_checkpoint)
                if size != last_size:
                    last_size = size
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"⏳ 训练中... (已运行 {elapsed/60:.1f} 分钟, "
                          f"检查点大小: {size/1024/1024:.1f} MB)")
            
            if os.path.exists(best_model):
                best_size = os.path.getsize(best_model)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"✅ 发现最佳模型 (大小: {best_size/1024/1024:.1f} MB)")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️ 监控已停止")
        print(f"⏱️ 总运行时间: {elapsed/60:.1f} 分钟")
        
        # 显示最终状态
        if os.path.exists(best_model):
            print(f"✅ 最佳模型已保存: {best_model}")
            print(f"   文件大小: {os.path.getsize(best_model)/1024/1024:.1f} MB")
        else:
            print("⚠️ 未找到最佳模型文件")

if __name__ == '__main__':
    monitor_training()
