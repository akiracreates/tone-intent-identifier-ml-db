SELECT
    p.intent_pred AS intent,
    COUNT(*) AS count,
    AVG(LENGTH(m.text)) AS avg_length_chars
FROM predictions p
JOIN messages m ON m.id = p.message_id
GROUP BY p.intent_pred
ORDER BY count DESC;
