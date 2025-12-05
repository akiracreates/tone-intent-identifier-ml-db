SELECT
    COUNT(*) AS total_predictions,
    MIN(predicted_at)  AS first_prediction_time,
    MAX(predicted_at) AS last_prediction_time
FROM predictions;
