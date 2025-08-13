import pandas as pd
import random

# Load the Excel file
def get_sentiments_by_counts(df):
    # file_path = 'C:/Users/PARUNRAT/Documents/BIS/Sentiment_analysis/Aldi/output_file.xlsx'  # Change this to your file path

    # # Read the Excel file into a DataFrame
    # df = pd.read_excel(file_path)
    # df = df.head(200)


    # Column names (replace with your actual column names)
    expected_date_col = 'Resolution expected until'  # Change this to the actual column name
    completed_date_col = 'Resolved'  # Change this to the actual column name
    start_date= 'Opened'
    # Calculate the difference and add a new column
    df['Days Late'] = (df[completed_date_col] - df[expected_date_col]).dt.days
    df['Days To Complete'] = (df[completed_date_col] - df[start_date]).dt.days
    df['Completed with Due'] = df.apply(lambda row: 1 if row[completed_date_col] > row[expected_date_col] and row['Sentiment']=='Neutral' else 0, axis=1)
    # df['Reassignment count > 3'] = df.apply(lambda row: 1 if row['Reassignment count'] > 3 and row['Completed with Due'] == 1 else 0, axis=1)
    df['Reopen count > 1'] = df.apply(lambda row: 1 if row['Reopen count'] > 1 and row['Completed with Due'] == 1 and row['Sentiment']=='Neutral' else 0, axis=1)

    # Add new columns for sub-category and category
    sub_categories = ['Prolonged Resolution Time', 'Missed Deadlines']
    df['Sentiment Sub-category'] = df.apply(
        lambda row: random.choice(sub_categories) if row['Completed with Due'] == 1 else row['Sentiment Sub-category'],
        axis=1
    )
    
    df['Sentiment Sub-category'] = df.apply(
        lambda row: "Incorrect Resolution" if row['Reopen count > 1'] == 1 and row['Completed with Due'] == 1 else row['Sentiment Sub-category'],
        axis=1
    )
    # For 'Category'
    df['Sentiment Category'] = df.apply(
        lambda row: 'Delayed Resolution' if row['Completed with Due'] == 1 else row['Sentiment Category'],
        axis=1
    )
    
    df['Sentiment Category'] = df.apply(
        lambda row: 'Delayed Resolution' if row['Reopen count > 1'] == 1 else row['Sentiment Category'],
        axis=1
    )
    df['Sentiment'] = df.apply(
        lambda row: 'Negative' if row['Sentiment Category'] == 'Delayed Resolution' else row['Sentiment'],
        axis=1
    )
    
    # Save the DataFrame back to an Excel file
    # output_file_path = 'C:/Users/PARUNRAT/Documents/BIS/Sentiment_analysis/Aldi/output_file.xlsx'  # Change this to your desired output file path
    # df.to_excel(output_file_path, index=False)

    return df
    # print(f"Processed data saved to {output_file_path}")