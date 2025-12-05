-- =========================================================
-- Schema for Tone & Intent Identifier: ML + DB Suite
-- Local PostgreSQL, course project
-- =========================================================

-- Drop tables if you need to reset (optional, for development)
-- WARNING: this deletes all data!
-- DROP TABLE IF EXISTS predictions;
-- DROP TABLE IF EXISTS messages;

-- -----------------------------
-- 1. Messages table
-- -----------------------------
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    true_tone VARCHAR(64),
    true_intent VARCHAR(64),
    created_at TIMESTAMP DEFAULT NOW()
);

-- -----------------------------
-- 2. Predictions table
-- -----------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    tone_bilstm_pred VARCHAR(64),
    tone_cnn_pred VARCHAR(64),
    intent_pred VARCHAR(64),
    predicted_at TIMESTAMP NOT NULL
);
