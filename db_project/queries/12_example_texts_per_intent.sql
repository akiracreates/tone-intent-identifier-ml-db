SELECT
    m.id,
    m.text,
    p.intent_pred,
    p.tone_bilstm_pred,
    p.tone_cnn_pred
FROM predictions p
JOIN messages m ON m.id = p.message_id
WHERE p.intent_pred = 'complaint'
LIMIT 20;
