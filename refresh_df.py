import pandas as pd

# Define the sentiment structure mapping
sentiment_structure = {
    "Positive": {
        "Quick Resolution": ["Quick Support", "Immediate Resolution", "Fast Turnaround", "SLA Adherence"],
        "Effective Communication": ["Clear and Open Communication", "Frequent Status Updates", "Transparency in Handling Tickets", "Acknowledgment with Gratitude"],
        "Friendly and Supportive Service": ["Friendly and Polite Support", "Respectful and Professional Team", "Supportive and Understanding Staff"],
        "Proactive and Personalized Support": ["Tailored Solutions", "Understanding Specific Needs", "Proactive Problem Identification", "Preemptive Action"],
        "Complete and Accurate Resolution": ["Comprehensive Fix", "Accurate Diagnosis", "Error-Free Resolution", "Efficient Problem Handling", "Empathy in Handling Cases", "Customer-Centric Approach"]
    },
    "Neutral": {
        "General Process Feedback": ["Standard Workflow Observations", "Suggestions for Improvement"],
        "Ambiguous Feedback": ["Non-Specific Comments", "Neutral Sentiments", "Generic Input"]
    },
    "Negative": {
        "Delayed Resolution": ["Missed Deadlines", "Slow Ticket Handling", "Prolonged Resolution Time", "Repeated Follow-Ups Required"],
        "Incomplete Solution": ["Persistent Problems", "Incomplete Fix", "Need for Further Investigation", "Recurring Issues", "Low Effort in Problem-Solving"],
        "Poor Communication": ["Lack of Updates", "Inconsistent Communication", "Confusing or Misleading Information", "Acknowledgment Without Resolution", "Waiting for Response or Updates", "Acknowledging Delays"],
        "Quality Below Expectations": ["Dismissive Attitude", "Lack of Courtesy", "Disrespectful Interactions", "Overall Poor Experience", "Service Quality Below Expectations", "Hard-to-Follow Procedures"],
        "No More Support Required": ["Resolved Without Support", "Customer-Fixed Issue", "Independent Problem Solving", "Abandoned Support Due to Ineffectiveness"]
    }
}

def update_sentiment_and_category(row):
    subcategory = row["Sentiment Sub-category"].strip().lower()  # Normalize to lowercase and strip whitespace

    for sentiment, categories in sentiment_structure.items():
        for category, subcategories in categories.items():
            normalized_subcategories = [s.lower() for s in subcategories]
            if subcategory in normalized_subcategories:
                return pd.Series({"Sentiment": sentiment, "Sentiment Category": category})

    return pd.Series({"Sentiment": None, "Sentiment Category": None})

# Read the Excel file into a DataFrame
df = pd.read_excel("C:/Users/PARUNRAT/Documents/BIS/Sentiment_analysis/Aldi/processed_feedback (8).xlsx")

# Apply the function to update both sentiment and sentiment category
df[["Sentiment", "Sentiment Category"]] = df.apply(update_sentiment_and_category, axis=1)

# Save the updated DataFrame to a new Excel file
df.to_excel('C:/Users/PARUNRAT/Documents/BIS/Sentiment_analysis/Aldi/output777.xlsx', index=False)

print("done")