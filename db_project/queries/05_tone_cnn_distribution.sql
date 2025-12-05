SELECT
    tone_cnn_pred AS tone,
    COUNT(*) AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM predictions
GROUP BY tone_cnn_pred
ORDER BY count DESC;
