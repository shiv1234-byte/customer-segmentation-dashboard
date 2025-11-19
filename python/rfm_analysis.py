import pandas as pd
from sklearn.cluster import KMeans

# Load data
customers = pd.read_csv('../data/customer_data.csv')
purchases = pd.read_csv('../data/purchase_data.csv')

# RFM Calculation
rfm = purchases.groupby('Customer_ID').agg({
    'Transaction_Date': lambda x: (pd.to_datetime('2025-12-31') - pd.to_datetime(x).max()).days,
    'Customer_ID': 'count',
    'Order_Value': 'sum'
}).rename(columns={
    'Transaction_Date': 'Recency',
    'Customer_ID': 'Frequency',
    'Order_Value': 'Monetary'
})

# Normalize RFM
rfm_norm = (rfm - rfm.min()) / (rfm.max() - rfm.min())

# K-Means Clustering (3 segments)
kmeans = KMeans(n_clusters=3, random_state=1)
rfm_norm['Segment'] = kmeans.fit_predict(rfm_norm)

# Output segmented customers
rfm_seg = rfm_norm.reset_index()
rfm_seg.to_csv('../data/rfm_segments.csv', index=False)
print(rfm_seg.head())
