-- SQLite 表结构（幂等，可重复执行）
-- 时间字段统一用 TEXT 存 ISO 字符串，与应用层现有写法一致，避免解析与时区漂移。
-- 从 MySQL 迁来的对应关系：JSON/LONGTEXT -> TEXT，DOUBLE -> REAL，
-- TINYINT(1) -> INTEGER，BIGINT AUTO_INCREMENT -> INTEGER PRIMARY KEY AUTOINCREMENT，
-- 内联 INDEX 拆成独立的 CREATE INDEX（SQLite 不支持写在表定义里）。

-- 足球预测+结算记录（result_sync）。记录为开放式 dict（业务会追加 hit flags、ml 评估等键），
-- 故用 promoted 列做查询 + doc 列存完整记录，保证读写完美往返、不丢键。
CREATE TABLE IF NOT EXISTS football_prediction (
    match_id    VARCHAR(128) PRIMARY KEY,
    league      VARCHAR(64),
    settled     INTEGER NOT NULL DEFAULT 0,
    sync_status VARCHAR(32),
    created_at  VARCHAR(64),
    updated_at  VARCHAR(64),
    doc         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_settled ON football_prediction (settled);
CREATE INDEX IF NOT EXISTS idx_sync_status ON football_prediction (sync_status);

-- 带模型版本的预测记录（prediction_records，用于版本对比/校准）。同样 promoted 列 + doc。
CREATE TABLE IF NOT EXISTS football_prediction_record (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id      VARCHAR(128),
    league        VARCHAR(64),
    model_version VARCHAR(64),
    created_at    VARCHAR(64),
    doc           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_match ON football_prediction_record (match_id);
CREATE INDEX IF NOT EXISTS idx_version ON football_prediction_record (model_version);

-- 球队 ELO 评分（当前值）
CREATE TABLE IF NOT EXISTS elo_rating (
    team       VARCHAR(128) PRIMARY KEY,
    rating     REAL NOT NULL,
    updated_at VARCHAR(64)
);

-- 球队 ELO 评分变更轨迹
CREATE TABLE IF NOT EXISTS elo_history (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    team   VARCHAR(128) NOT NULL,
    rating REAL,
    date   VARCHAR(64),
    event  VARCHAR(64)
);
CREATE INDEX IF NOT EXISTS idx_team ON elo_history (team);

-- 相似盘口样本库（原 8MB JSON，拆为行）
CREATE TABLE IF NOT EXISTS similar_market (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asian           REAL,
    asian_odds_home REAL,
    asian_odds_away REAL,
    total           REAL,
    total_over      REAL,
    total_under     REAL,
    euro_home       REAL,
    euro_draw       REAL,
    euro_away       REAL,
    result          VARCHAR(8),
    goals_home      INTEGER,
    goals_away      INTEGER,
    date            VARCHAR(32),
    league          VARCHAR(32),
    home_team       VARCHAR(128),
    away_team       VARCHAR(128)
);
CREATE INDEX IF NOT EXISTS idx_market ON similar_market (asian, total);

-- 通用键值存储：配置、统计 blob、每日缓存等形态杂的数据
-- cache_date 非空表示这是带每日失效语义的缓存项。
CREATE TABLE IF NOT EXISTS kv_store (
    k          VARCHAR(128) PRIMARY KEY,
    json_value TEXT,
    cache_date VARCHAR(16),
    updated_at VARCHAR(64)
);

-- 比赛历史全量库（football-data.co.uk CSV 导入，match_id 幂等）
-- match_date 存 'YYYY-MM-DD'、match_time 存 'HH:MM:SS'，
-- iter_csv_rows 的 strftime 按这个格式还原 CSV 列。
CREATE TABLE IF NOT EXISTS matches (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id     VARCHAR(256) UNIQUE NOT NULL,
    league       VARCHAR(64),
    league_code  VARCHAR(8),
    match_date   TEXT,
    match_time   TEXT,
    home_team    VARCHAR(128),
    away_team    VARCHAR(128),
    fthg         INTEGER,
    ftag         INTEGER,
    ftr          VARCHAR(1),
    hthg         INTEGER,
    htag         INTEGER,
    htr          VARCHAR(1),
    odds         TEXT,
    stats        TEXT,
    settled      INTEGER DEFAULT 0,
    created_at   VARCHAR(64),
    updated_at   VARCHAR(64)
);
CREATE INDEX IF NOT EXISTS idx_matches_league ON matches (league_code);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches (match_date);
CREATE INDEX IF NOT EXISTS idx_matches_home ON matches (home_team);
CREATE INDEX IF NOT EXISTS idx_matches_away ON matches (away_team);
