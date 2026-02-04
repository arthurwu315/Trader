-- MVP Gate數據庫遷移
-- 新增gate_decisions表

-- 創建gate_decisions表
CREATE TABLE IF NOT EXISTS gate_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    decision TEXT NOT NULL,  -- 'ALLOW' or 'REJECT'
    reason_code TEXT NOT NULL,
    
    -- 帳戶狀態
    available_balance REAL,
    wallet_balance REAL,
    margin_ratio REAL,
    
    -- 交易信息
    notional REAL,
    required_margin REAL,
    risk_usdt REAL,
    
    -- Debug信息
    debug_json TEXT,
    
    -- 索引
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 創建索引
CREATE INDEX IF NOT EXISTS idx_gate_timestamp ON gate_decisions(timestamp);
CREATE INDEX IF NOT EXISTS idx_gate_decision ON gate_decisions(decision);
CREATE INDEX IF NOT EXISTS idx_gate_reason ON gate_decisions(reason_code);
CREATE INDEX IF NOT EXISTS idx_gate_symbol ON gate_decisions(symbol);

-- 創建拒單統計視圖
CREATE VIEW IF NOT EXISTS gate_reject_stats AS
SELECT 
    reason_code,
    COUNT(*) as count,
    AVG(available_balance) as avg_available,
    AVG(margin_ratio) as avg_margin_ratio,
    MAX(timestamp) as last_reject
FROM gate_decisions
WHERE decision = 'REJECT'
GROUP BY reason_code
ORDER BY count DESC;

-- 創建通過率視圖
CREATE VIEW IF NOT EXISTS gate_pass_rate AS
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total,
    SUM(CASE WHEN decision = 'ALLOW' THEN 1 ELSE 0 END) as allowed,
    SUM(CASE WHEN decision = 'REJECT' THEN 1 ELSE 0 END) as rejected,
    CAST(SUM(CASE WHEN decision = 'ALLOW' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) * 100 as pass_rate
FROM gate_decisions
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- 查詢今日拒單原因
-- SELECT reason_code, COUNT(*) 
-- FROM gate_decisions 
-- WHERE decision = 'REJECT' AND DATE(timestamp) = DATE('now')
-- GROUP BY reason_code;

-- 查詢最近10次Gate決策
-- SELECT timestamp, decision, reason_code, available_balance, margin_ratio
-- FROM gate_decisions
-- ORDER BY id DESC
-- LIMIT 10;
