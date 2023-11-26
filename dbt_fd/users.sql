SELECT *
FROM {{ source("recommender_system_raw", "users") }}
