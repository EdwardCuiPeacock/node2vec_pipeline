-- SQL query string to load the graph data
-- Jinja2 templated
-- * GOOGLE_CLOUD_PROJECT: GCP Bigquery database project name
WITH titles AS (
    SELECT DISTINCT COALESCE(InSeasonSeries_Id, TitleId) as InSeasonSeries_Id,
        TitleDetails_LongSynopsis,
        InSeasonSeries_Tags,
        TitleTags,
        TitleSubgenres,
        TitleType
    FROM `{{ GOOGLE_CLOUD_PROJECT }}.recsystem.ContentMetadataView`
),
melted AS (
    SELECT DISTINCT InSeasonSeries_Id,
        TitleDetails_LongSynopsis,
        TitleType,
        TRIM(tags) as tags
    FROM (
            SELECT DISTINCT InSeasonSeries_Id,
                TitleDetails_LongSynopsis,
                TitleType,
                tags
            FROM titles
                CROSS JOIN UNNEST(SPLIT(InSeasonSeries_Tags, ',')) tags
            UNION ALL
            SELECT DISTINCT InSeasonSeries_Id,
                TitleDetails_LongSynopsis,
                TitleType,
                tags
            FROM titles
                CROSS JOIN UNNEST(SPLIt(TitleSubgenres, ',')) tags
            UNION ALL
            SELECT DISTINCT InSeasonSeries_Id,
                TitleDetails_LongSynopsis,
                TitleType,
                tags
            FROM titles
                CROSS JOIN UNNEST(SPLIt(TitleTags, ',')) tags
        )
    WHERE tags <> ''
),
-- get tokens from long synopsis
token_table AS (
    SELECT COALESCE(InSeasonSeries_Id, TitleId) AS InSeasonSeries_Id,
        SPLIT(
            REGEXP_REPLACE(
                LOWER(TitleDetails_LongSynopsis),
                '[^a-zA-Z0-9 -]',
                ''
            ),
            ' '
        ) AS tokens,
        -- filter out non-alphabetical characters
    FROM `{{ GOOGLE_CLOUD_PROJECT }}.recsystem.ContentMetadataView`
),
-- unnest token
token_clean AS (
    SELECT InSeasonSeries_Id,
        token,
        COUNT(*) AS token_count
    FROM token_table
        CROSS JOIN UNNEST(tokens) token
    GROUP BY InSeasonSeries_Id,
        token
)
SELECT distinct InSeasonSeries_Id,
    tags AS token,
    1 AS token_count
FROM melted
UNION ALL
(
    SELECT InSeasonSeries_Id,
        token,
        token_count
    FROM token_clean t
        LEFT OUTER JOIN `{{ GOOGLE_CLOUD_PROJECT }}.recsystem.stop_words_en_sp` stop ON stop.string_field_0 = t.token
    WHERE stop.string_field_0 IS NULL
)
LIMIT 100