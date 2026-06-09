"""
人脸特征数据库操作模块

功能：
  - 注册/删除/更新/查询 人脸特征
  - 批量导入导出
  - 识别日志记录
  - 用户信息管理
"""

import numpy as np
import pymysql

# ── 默认数据库配置 ──────────────────────────────────────────

DEFAULT_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "004520HHA",
    "database": "face_system",
    "charset": "utf8mb4",
}


class FaceDatabase:
    """人脸特征数据库，BLOB 存储 + 向量化搜索。"""

    def __init__(self, **kwargs):
        config = {**DEFAULT_CONFIG, **kwargs}
        self.conn = pymysql.connect(**config)
        self.cursor = self.conn.cursor()
        self._init_tables()

    # ── 表初始化 ──────────────────────────────────────────

    def _init_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_features (
                id          INT AUTO_INCREMENT PRIMARY KEY,
                user_id     VARCHAR(50)  UNIQUE NOT NULL,
                name        VARCHAR(100) NOT NULL,
                feature_vec LONGBLOB NOT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          INT AUTO_INCREMENT PRIMARY KEY,
                user_id     VARCHAR(50)  UNIQUE NOT NULL,
                name        VARCHAR(100) NOT NULL,
                gender      VARCHAR(10)  DEFAULT '其他',
                phone       VARCHAR(20)  DEFAULT '',
                department  VARCHAR(100) DEFAULT '',
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id          INT AUTO_INCREMENT PRIMARY KEY,
                user_id     VARCHAR(50)  DEFAULT NULL,
                name        VARCHAR(100) DEFAULT NULL,
                similarity  FLOAT DEFAULT 0,
                snapshot     VARCHAR(500) DEFAULT '',
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

        # 自动添加 photo_path 列（兼容旧表）
        for table in ['face_features', 'users']:
            try:
                self.cursor.execute(
                    f"ALTER TABLE `{table}` ADD COLUMN photo_path VARCHAR(500) DEFAULT ''"
                )
                self.conn.commit()
            except pymysql.err.OperationalError:
                pass  # 列已存在，忽略

    # ── 注册 ──────────────────────────────────────────────

    def register(self, user_id, name, feature_vec, gender="其他", phone="", department=""):
        """注册用户并存储人脸特征。

        Args:
            user_id: 用户唯一标识
            name: 姓名
            feature_vec: numpy array, shape (512,)
            gender: 性别
            phone: 电话
            department: 部门/班级
        """
        blob = feature_vec.astype(np.float32).tobytes()
        # 写入特征表
        self.cursor.execute(
            "REPLACE INTO face_features (user_id, name, feature_vec) VALUES (%s, %s, %s)",
            (user_id, name, blob),
        )
        # 同步写入用户表
        self.cursor.execute(
            """INSERT INTO users (user_id, name, gender, phone, department)
               VALUES (%s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE name=%s, gender=%s, phone=%s, department=%s""",
            (user_id, name, gender, phone, department, name, gender, phone, department),
        )
        self.conn.commit()

    def register_with_photo(self, user_id, name, feature_vec, photo_path,
                            gender="其他", phone="", department=""):
        """注册用户并同时写入照片路径（原子事务）。

        Args:
            user_id: 用户唯一标识
            name: 姓名
            feature_vec: numpy array, shape (512,)
            photo_path: 头像照片保存路径
            gender: 性别
            phone: 电话
            department: 部门/班级
        """
        blob = feature_vec.astype(np.float32).tobytes()
        self.cursor.execute(
            "REPLACE INTO face_features (user_id, name, feature_vec, photo_path) VALUES (%s, %s, %s, %s)",
            (user_id, name, blob, photo_path),
        )
        self.cursor.execute(
            """INSERT INTO users (user_id, name, gender, phone, department, photo_path)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE name=%s, gender=%s, phone=%s, department=%s, photo_path=%s""",
            (user_id, name, gender, phone, department, photo_path,
             name, gender, phone, department, photo_path),
        )
        self.conn.commit()

    def register_batch(self, records):
        """批量注册（records: list of dict，每个 dict 含 user_id, name, feature_vec）。

        也支持旧格式 list of (user_id, name, feature_vec)。
        """
        for rec in records:
            if isinstance(rec, (list, tuple)):
                uid, name, feat = rec
                self.register(uid, name, feat)
            elif isinstance(rec, dict):
                self.register(
                    user_id=rec["user_id"],
                    name=rec["name"],
                    feature_vec=rec["feature_vec"],
                    gender=rec.get("gender", "其他"),
                    phone=rec.get("phone", ""),
                    department=rec.get("department", ""),
                )

    def register_multi_features(self, user_id, name, feature_list, **kwargs):
        """为同一用户注册多张人脸特征（取平均后存储，提高鲁棒性）。

        Args:
            feature_list: list of numpy arrays, each shape (512,)
        """
        avg_feat = np.mean(feature_list, axis=0)
        self.register(user_id, name, avg_feat, **kwargs)

    # ── 查询 ──────────────────────────────────────────────

    def load_all_features(self):
        """加载全部特征到 numpy 矩阵 → (features (N,512), ids list)。"""
        self.cursor.execute("SELECT user_id, feature_vec FROM face_features")
        rows = self.cursor.fetchall()
        if not rows:
            return np.empty((0, 512), dtype=np.float32), []
        ids = [r[0] for r in rows]
        features = np.array(
            [np.frombuffer(r[1], dtype=np.float32) for r in rows], dtype=np.float32
        )
        return features, ids

    def load_all_features_with_names(self):
        """加载全部特征 + 姓名 → (features (N,512), ids list, names list)。"""
        self.cursor.execute("SELECT user_id, name, feature_vec FROM face_features")
        rows = self.cursor.fetchall()
        if not rows:
            return np.empty((0, 512), dtype=np.float32), [], []
        ids = [r[0] for r in rows]
        names = [r[1] for r in rows]
        features = np.array(
            [np.frombuffer(r[2], dtype=np.float32) for r in rows], dtype=np.float32
        )
        return features, ids, names

    def get_feature(self, user_id):
        """获取指定用户的特征向量。"""
        self.cursor.execute(
            "SELECT feature_vec FROM face_features WHERE user_id=%s", (user_id,)
        )
        row = self.cursor.fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32)

    def get_user_info(self, user_id=None, name=None):
        """查询用户信息。

        优先按 user_id 查找，其次按 name 模糊匹配。
        """
        if user_id:
            self.cursor.execute(
                "SELECT * FROM users WHERE user_id=%s", (user_id,)
            )
        elif name:
            self.cursor.execute(
                "SELECT * FROM users WHERE name LIKE %s", (f"%{name}%",)
            )
        else:
            return []
        rows = self.cursor.fetchall()
        cols = [desc[0] for desc in self.cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    def list_users(self):
        """列出所有已注册用户（不含特征向量）。"""
        self.cursor.execute(
            "SELECT user_id, name, gender, phone, department, COALESCE(photo_path,'') as photo_path, "
            "created_at FROM users ORDER BY created_at DESC"
        )
        cols = [desc[0] for desc in self.cursor.description]
        return [dict(zip(cols, row)) for row in self.cursor.fetchall()]

    # ── 搜索 / 识别 ──────────────────────────────────────

    def search(self, query_vec, threshold=0.45):
        """向量化 1:N 搜索（一次 np.dot 完成全库匹配）。

        Returns:
            (match_info, max_sim)
            match_info: (user_id, similarity) 或 None
        """
        features, ids = self.load_all_features()
        if len(ids) == 0:
            return None, 0.0
        sims = features @ query_vec
        idx = int(np.argmax(sims))
        max_sim = float(sims[idx])
        if max_sim > threshold:
            return (ids[idx], max_sim), max_sim
        return None, max_sim

    def search_with_name(self, query_vec, threshold=0.45):
        """搜索并返回姓名。

        Returns:
            (user_id, name, similarity) 或 (None, None, similarity)
        """
        features, ids, names = self.load_all_features_with_names()
        if len(ids) == 0:
            return None, None, 0.0
        sims = features @ query_vec
        idx = int(np.argmax(sims))
        max_sim = float(sims[idx])
        if max_sim > threshold:
            return ids[idx], names[idx], max_sim
        return None, None, max_sim

    def get_gallery(self):
        """获取完整 gallery 列表，供 FaceEngine.identify() 使用。"""
        self.cursor.execute("SELECT user_id, name, feature_vec FROM face_features")
        return [
            (uid, name, np.frombuffer(blob, dtype=np.float32))
            for uid, name, blob in self.cursor.fetchall()
        ]

    # ── 更新 ──────────────────────────────────────────────

    def update_user(self, user_id, name=None, gender=None, phone=None, department=None):
        """更新用户信息（不含特征向量）。"""
        fields, values = [], []
        if name is not None:
            fields.append("name=%s")
            values.append(name)
        if gender is not None:
            fields.append("gender=%s")
            values.append(gender)
        if phone is not None:
            fields.append("phone=%s")
            values.append(phone)
        if department is not None:
            fields.append("department=%s")
            values.append(department)
        if not fields:
            return
        values.append(user_id)
        sql = f"UPDATE users SET {', '.join(fields)} WHERE user_id=%s"
        self.cursor.execute(sql, tuple(values))
        # 同步更新 face_features 中的 name
        if name is not None:
            self.cursor.execute(
                "UPDATE face_features SET name=%s WHERE user_id=%s", (name, user_id)
            )
        self.conn.commit()

    def update_feature(self, user_id, new_feature):
        """更新用户的人脸特征向量。"""
        blob = new_feature.astype(np.float32).tobytes()
        self.cursor.execute(
            "UPDATE face_features SET feature_vec=%s WHERE user_id=%s", (blob, user_id)
        )
        self.conn.commit()

    def append_feature(self, user_id, new_feature, alpha=0.3):
        """增量更新特征：new_feat = alpha * new + (1-alpha) * old（平滑融合）。"""
        old = self.get_feature(user_id)
        if old is None:
            return self.register(user_id, "", new_feature)
        merged = alpha * new_feature + (1 - alpha) * old
        merged = merged / (np.linalg.norm(merged) + 1e-8)  # 重新归一化
        self.update_feature(user_id, merged)

    # ── 删除 ──────────────────────────────────────────────

    def delete_user(self, user_id):
        """删除用户（同时删除特征和日志）。"""
        self.cursor.execute("DELETE FROM face_features WHERE user_id=%s", (user_id,))
        self.cursor.execute("DELETE FROM users WHERE user_id=%s", (user_id,))
        self.cursor.execute("DELETE FROM recognition_logs WHERE user_id=%s", (user_id,))
        self.conn.commit()

    def clear_all(self):
        """清空所有数据（慎用！）。"""
        self.cursor.execute("DELETE FROM face_features")
        self.cursor.execute("DELETE FROM users")
        self.cursor.execute("DELETE FROM recognition_logs")
        self.conn.commit()

    # ── 识别日志 ──────────────────────────────────────────

    def log_recognition(self, user_id, name, similarity, snapshot=""):
        """记录一次识别结果。"""
        self.cursor.execute(
            """INSERT INTO recognition_logs (user_id, name, similarity, snapshot)
               VALUES (%s, %s, %s, %s)""",
            (user_id, name, similarity, snapshot),
        )
        self.conn.commit()

    def get_logs(self, limit=50, offset=0):
        """获取最近的识别日志。"""
        self.cursor.execute(
            "SELECT * FROM recognition_logs ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (limit, offset),
        )
        cols = [desc[0] for desc in self.cursor.description]
        return [dict(zip(cols, row)) for row in self.cursor.fetchall()]

    def get_attendance_logs(self, date_from=None, date_to=None, limit=500):
        """获取考勤记录（支持日期筛选）。

        Args:
            date_from: 起始日期字符串 'YYYY-MM-DD'，默认 7 天前
            date_to:   截止日期字符串 'YYYY-MM-DD'，默认今天
            limit:     最大返回条数
        """
        self.cursor.execute(
            "SELECT recognition_logs.*, users.gender, users.phone, users.department "
            "FROM recognition_logs "
            "LEFT JOIN users ON recognition_logs.user_id = users.user_id "
            "WHERE recognition_logs.created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) "
            "AND recognition_logs.created_at < CURDATE() + INTERVAL 1 DAY "
            "ORDER BY recognition_logs.created_at DESC LIMIT %s",
            (limit,),
        )
        cols = [desc[0] for desc in self.cursor.description]
        return [dict(zip(cols, row)) for row in self.cursor.fetchall()]

    def get_attendance_stats(self, date_from=None, date_to=None):
        """获取考勤统计：每人每日首次识别视为到勤。

        Args:
            date_from: 起始日期字符串 'YYYY-MM-DD'
            date_to:   截止日期字符串 'YYYY-MM-DD'

        Returns:
            list of dict: name, att_date, first_time, rec_count
        """
        if date_from and date_to:
            self.cursor.execute(
                """SELECT name, DATE(created_at) as att_date,
                           MIN(TIME(created_at)) as first_time,
                           COUNT(*) as rec_count
                    FROM recognition_logs
                    WHERE created_at >= %s AND created_at < %s + INTERVAL 1 DAY
                    GROUP BY name, DATE(created_at)
                    ORDER BY att_date DESC, first_time ASC""",
                (date_from, date_to),
            )
        elif date_from:
            self.cursor.execute(
                """SELECT name, DATE(created_at) as att_date,
                           MIN(TIME(created_at)) as first_time,
                           COUNT(*) as rec_count
                    FROM recognition_logs
                    WHERE created_at >= %s
                    GROUP BY name, DATE(created_at)
                    ORDER BY att_date DESC, first_time ASC""",
                (date_from,),
            )
        else:
            self.cursor.execute(
                """SELECT name, DATE(created_at) as att_date,
                           MIN(TIME(created_at)) as first_time,
                           COUNT(*) as rec_count
                    FROM recognition_logs
                    WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
                    AND created_at < CURDATE() + INTERVAL 1 DAY
                    GROUP BY name, DATE(created_at)
                    ORDER BY att_date DESC, first_time ASC"""
            )
        cols = [desc[0] for desc in self.cursor.description]
        return [dict(zip(cols, row)) for row in self.cursor.fetchall()]

    def clear_logs(self):
        """清空识别日志。"""
        self.cursor.execute("DELETE FROM recognition_logs")
        self.conn.commit()

    # ── 统计 ──────────────────────────────────────────────

    def count(self):
        """已注册人脸数量。"""
        self.cursor.execute("SELECT COUNT(*) FROM face_features")
        return self.cursor.fetchone()[0]

    def count_logs(self):
        """识别日志数量。"""
        self.cursor.execute("SELECT COUNT(*) FROM recognition_logs")
        return self.cursor.fetchone()[0]

    # ── 关闭 ──────────────────────────────────────────────

    def close(self):
        self.cursor.close()
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
