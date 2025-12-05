SELECT
    m.id,
    m.text,
    p.tone_bilstm_pred,
    p.tone_cnn_pred,
    p.intent_pred,
    p.predicted_at
FROM predictions p
JOIN messages m ON m.id = p.message_id
WHERE p.tone_bilstm_pred <> p.tone_cnn_pred
ORDER BY p.predicted_at
LIMIT 50;