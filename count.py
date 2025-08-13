import pandas as pd

# Load the Excel file
def get_count_sentiments(file_path):
    # file_path =''  # Replace with your file path
    # sheet_name = 'Sheet1'  # Replace with your sheet name if different

    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    # Group by 'Assignment Group', 'Sentiment', 'Category', and 'Sub-category'
    grouped_data = df.groupby(['Impacted Service', 'Sentiment', 'Sentiment Category', 'Sentiment Sub-category']).size().reset_index(name='Count')

    # Create a summary text
    summary_text = ""
    cnt=0
    for assignment_group in grouped_data['Impacted Service'].unique():
        summary_text += f"\nImpacted Service: {assignment_group}\n"
        assignment_group_data = grouped_data[grouped_data['Impacted Service'] == assignment_group]
        # print(assignment_group)
        for sentiment in ['Positive', 'Negative']:
            summary_text += f"  {sentiment} Sentiments:\n"
            sentiment_data = assignment_group_data[assignment_group_data['Sentiment'] == sentiment]
            # print(sentiment)
            for category in sentiment_data['Sentiment Category'].unique():
                summary_text += f"   Sentiment Category: {category}\n"
                category_data = sentiment_data[sentiment_data['Sentiment Category'] == category]
                # print(category)
                for sub_category in category_data['Sentiment Sub-category'].unique():
                    sub_category_count = category_data[category_data['Sentiment Sub-category'] == sub_category]['Count'].sum()
                    summary_text += f"     Sentiment Sub-category: {sub_category} - Count: {sub_category_count}\n"
                    cnt+=sub_category_count
                    # print(sub_category,":",sub_category_count)
    # Print the summary text
    # print(cnt)
    return summary_text


# txt=get_count_sentiments("C:/Users/PARUNRAT/Documents/BIS/Sentiment_analysis/Aldi/processed_feedback (8).xlsx")    
# print(txt)
