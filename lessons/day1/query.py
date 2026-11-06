import os

from google.cloud import bigquery

bq_client = bigquery.Client(project=os.environ.get("PROJECT_ID"), location=os.environ.get("REGION"))

q = """
SELECT 
    INDUSTRY,
    ROUND(SUM(CLAIMS_Y1 + CLAIMS_Y2 + CLAIMS_Y3) / SUM(PREMIUM_Y1 + PREMIUM_Y2 + PREMIUM_Y3), 3) AS LOSS_RATIO
FROM `eb_rampup.groups`
GROUP BY INDUSTRY
ORDER BY LOSS_RATIO DESC
"""
qdf = bq_client.query(q).to_dataframe()

print(qdf)
