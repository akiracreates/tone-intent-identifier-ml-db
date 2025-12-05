SELECT
    AVG(LENGTH(m.text)) AS avg_length_chars,
    MIN(LENGTH(m.text)) AS min_length_chars,
    MAX(LENGTH(m.text)) AS max_length_chars,
    AVG(array_length(regexp_split_to_array(m.text, E'\s+'), 1)) AS avg_length_words
FROM messages m;
