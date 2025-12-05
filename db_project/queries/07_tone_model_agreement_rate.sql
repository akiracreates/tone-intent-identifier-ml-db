SELECT
    SUM(CASE WHEN tone_bilstm_pred = tone_cnn_pred THEN 1 ELSE 0 END) AS agree_count,
    SUM(CASE WHEN tone_bilstm_pred <> tone_cnn_pred THEN 1 ELSE 0 END) AS disagree_count,
    ROUND(
        100.0 * SUM(CASE WHEN tone_bilstm_pred = tone_cnn_pred THEN 1 ELSE 0 END)
        / COUNT(*),
        2
    ) AS agreement_percent
FROM predictions;
