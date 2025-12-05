SELECT
    tone_bilstm_pred AS bilstm_tone,
    tone_cnn_pred AS cnn_tone,
    COUNT(*) AS count
FROM predictions
WHERE tone_bilstm_pred <> tone_cnn_pred
GROUP BY tone_bilstm_pred, tone_cnn_pred
ORDER BY count DESC;