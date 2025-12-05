SELECT
    p.intent_pred AS intent,
    p.tone_bilstm_pred AS tone,
    COUNT(*) AS count
FROM predictions p
GROUP BY p.intent_pred, p.tone_bilstm_pred
ORDER BY intent, tone;
