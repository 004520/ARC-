"""
数据库初始化脚本 — 一键创建 face_system 数据库及所有表

用法：python init_db.py
"""

import pymysql

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "004520HHA",
    "charset": "utf8mb4",
}

TARGET_DB = "face_system"


def init_database():
    # 先不指定 db，连接 MySQL 服务本身
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 1) 创建数据库（如果不存在）
    cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS `{TARGET_DB}` "
        f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    print(f"[OK] 数据库 `{TARGET_DB}` 已就绪")

    # 2) 切换到目标库
    cursor.execute(f"USE `{TARGET_DB}`")

    # 3) 创建用户表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS `users` (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            user_id     VARCHAR(50)  UNIQUE NOT NULL COMMENT '用户唯一标识',
            name        VARCHAR(100) NOT NULL COMMENT '姓名',
            gender      VARCHAR(10) DEFAULT '其他',
            phone       VARCHAR(20) DEFAULT '',
            department  VARCHAR(100) DEFAULT '' COMMENT '部门/班级',
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户基本信息'
    """)

    # 4) 创建人脸特征表（原表，保留兼容）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS `face_features` (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            user_id     VARCHAR(50) UNIQUE NOT NULL,
            name        VARCHAR(100) NOT NULL,
            feature_vec LONGBLOB NOT NULL COMMENT '512维 float32 特征向量',
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='人脸特征向量'
    """)

    # 5) 创建识别日志表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS `recognition_logs` (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            user_id     VARCHAR(50) DEFAULT NULL,
            name        VARCHAR(100) DEFAULT NULL,
            similarity  FLOAT DEFAULT 0,
            snapshot     VARCHAR(500) DEFAULT '' COMMENT '抓拍图片路径',
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='识别记录日志'
    """)

    conn.commit()

    # 打印表统计
    for table in ["users", "face_features", "recognition_logs"]:
        cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
        cnt = cursor.fetchone()[0]
        print(f"  表 `{table}`: {cnt} 条记录")

    cursor.close()
    conn.close()
    print("\n[完成] 数据库初始化成功，可以开始使用！")


if __name__ == "__main__":
    init_database()
