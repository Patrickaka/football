-- MySQL 表结构（幂等，可重复执行）
-- 时间字段统一用 VARCHAR 存 ISO 字符串，与应用层现有写法一致，避免解析与时区漂移。

-- 足球预测+结算记录（result_sync）。记录为开放式 dict（业务会追加 hit flags、ml 评估等键），
-- 故用 promoted 列做查询 + doc 列存完整记录，保证读写完美往返、不丢键。
CREATE TABLE IF NOT EXISTS football_prediction (
    match_id    VARCHAR(128) PRIMARY KEY,
    league      VARCHAR(64),
    settled     TINYINT(1) NOT NULL DEFAULT 0,
    sync_status VARCHAR(32),
    created_at  VARCHAR(64),
    updated_at  VARCHAR(64),
    doc         JSON NOT NULL,
    INDEX idx_settled (settled),
    INDEX idx_sync_status (sync_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 带模型版本的预测记录（prediction_records，用于版本对比/校准）。同样 promoted 列 + doc。
CREATE TABLE IF NOT EXISTS football_prediction_record (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    match_id      VARCHAR(128),
    league        VARCHAR(64),
    model_version VARCHAR(64),
    created_at    VARCHAR(64),
    doc           JSON NOT NULL,
    INDEX idx_match (match_id),
    INDEX idx_version (model_version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 大乐透开奖历史。seq 自增主键保留原列表顺序（业务依赖 index 0 为最新）。
CREATE TABLE IF NOT EXISTS dlt_history (
    seq       BIGINT AUTO_INCREMENT PRIMARY KEY,
    issue     VARCHAR(32),
    front     JSON,
    back      JSON,
    draw_date VARCHAR(32),
    INDEX idx_issue (issue)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 排列5开奖历史。seq 自增主键保留原列表顺序。
CREATE TABLE IF NOT EXISTS pailie5_history (
    seq       BIGINT AUTO_INCREMENT PRIMARY KEY,
    issue     VARCHAR(32),
    numbers   JSON,
    draw_date VARCHAR(32),
    ts        VARCHAR(64),
    INDEX idx_issue (issue)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 球队 ELO 评分（当前值）
CREATE TABLE IF NOT EXISTS elo_rating (
    team       VARCHAR(128) PRIMARY KEY,
    rating     DOUBLE NOT NULL,
    updated_at VARCHAR(64)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 球队 ELO 评分变更轨迹
CREATE TABLE IF NOT EXISTS elo_history (
    id     BIGINT AUTO_INCREMENT PRIMARY KEY,
    team   VARCHAR(128) NOT NULL,
    rating DOUBLE,
    date   VARCHAR(64),
    event  VARCHAR(64),
    INDEX idx_team (team)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 相似盘口样本库（原 8MB JSON，拆为行）
CREATE TABLE IF NOT EXISTS similar_market (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    asian           DOUBLE,
    asian_odds_home DOUBLE,
    asian_odds_away DOUBLE,
    total           DOUBLE,
    total_over      DOUBLE,
    total_under     DOUBLE,
    euro_home       DOUBLE,
    euro_draw       DOUBLE,
    euro_away       DOUBLE,
    result          VARCHAR(8),
    goals_home      INT,
    goals_away      INT,
    date            VARCHAR(32),
    league          VARCHAR(32),
    home_team       VARCHAR(128),
    away_team       VARCHAR(128),
    INDEX idx_market (asian, total)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 通用键值存储：配置、统计 blob、每日缓存等形态杂的数据
-- cache_date 非空表示这是带每日失效语义的缓存项。
CREATE TABLE IF NOT EXISTS kv_store (
    k          VARCHAR(128) PRIMARY KEY,
    json_value LONGTEXT,
    cache_date VARCHAR(16),
    updated_at VARCHAR(64)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 比赛历史全量库（football-data.co.uk CSV 导入，match_id 幂等）
CREATE TABLE IF NOT EXISTS matches (
    id           BIGINT AUTO_INCREMENT PRIMARY KEY,
    match_id     VARCHAR(256) UNIQUE NOT NULL,
    league       VARCHAR(64),
    league_code  VARCHAR(8),
    match_date   DATE,
    match_time   TIME NULL,
    home_team    VARCHAR(128),
    away_team    VARCHAR(128),
    fthg         INT,
    ftag         INT,
    ftr          VARCHAR(1),
    hthg         INT,
    htag         INT,
    htr          VARCHAR(1),
    odds         JSON,
    stats        JSON,
    settled      TINYINT(1) DEFAULT 0,
    created_at   VARCHAR(64),
    updated_at   VARCHAR(64),
    INDEX idx_matches_league (league_code),
    INDEX idx_matches_date (match_date),
    INDEX idx_matches_home (home_team),
    INDEX idx_matches_away (away_team)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
