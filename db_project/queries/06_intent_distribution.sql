SELECT
    intent_pred AS intent,
    COUNT(*) AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM predictions
GROUP BY intent_pred
ORDER BY count DESC;
