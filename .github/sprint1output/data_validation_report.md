# Data Validation Report

Rows validated: 100,000

- amt <= 0: 0
- Customer_Age outside 0-120: 0
- Customer_Satisfaction_Score outside 1-10: 0
- lat outside -90..90: 0
- long outside -180..180: 0
- is_fraud not in {0,1}: 0
- gender not in {M,F}: 0
- duplicate trans_num: 0
- Transaction_Time matches real timestamp (rate): 0.0007
- Customer_Age matches dob-derived age (rate): 0.017

**Note:** `Transaction_Time`, `Customer_Age`, and `Merchant_Category` show very low agreement with the real transaction timestamp / dob-derived age / true category, meaning they appear to be randomly generated rather than derived fields. They should not be trusted for analysis or modeling.
