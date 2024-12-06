#### THis will give you the top 4 dicoms with avg pixel value, you can adjust that number as needed

import pandas as pd

df = pd.read_csv('OGids_with_metrics.csv')
# Count unique study_id values
unique_study_ids = df['study_id'].nunique()

# Print the result
print("Number of unique study_id values:", unique_study_ids)

df = df.sort_values(by=['study_id', 'avg_pixel_value'], ascending=[True, False])

# Group by 'study_id' and keep only the top 4 rows with the highest 'avg_pixel_value' in each group   Adjust as needed
df_top4 = df.groupby('study_id').head(4).reset_index(drop=True)

# Count unique study_id values
unique_study_ids4 = df_top4['study_id'].nunique()

# Print the result
print("Number of unique study_id values:", unique_study_ids4)

df_top4.to_csv('ogtop4select.csv')