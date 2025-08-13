import streamlit as st
import pandas as pd
import openai
from io import BytesIO
from due import get_sentiments_by_counts
import os
from dotenv import load_dotenv
load_dotenv()
client = openai.AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT")
)

st.title("🔍 BEAT - Business Emotion Assessment Tracker")

uploaded_file = st.file_uploader("📤 Upload an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = df.head(10)  # Limit to first 100 rows for performance
    # Filter rows where 'u_smiley_survey_response' is not NaN and contains strings
    # df = df.dropna(subset=['u_smiley_survey_response'])
    # df = df[df['u_smiley_survey_response'].apply(lambda x: isinstance(x, str))]

    # Display the filtered DataFrame
    # st.write(df)
    # st.write(len(df))
    # if 'u_smiley_survey_rating' not in df.columns:
    #     st.warning("⚠️ 'u_smiley_survey_rating' column not found. Please make sure your Excel file contains this column.")
    # else:
    #     df['Rating'] = df['u_smiley_survey_rating']  # Keep a copy for output

    st.subheader("📌 Select Feedback Column")
    selected_column = st.selectbox("Choose the column containing feedback text:", df.columns)
    df = df.dropna(subset=[selected_column])

    def batch_analyze(feedback_list,batch_size_sending):
        combined_feedback = "\n".join(
            [f"{i+1}. Feedback: {feedback_list[i].strip().lower()}" for i in range(len(feedback_list))]
        )

        # Define prompt
        prompt = f"""
        You are an intelligent assistant. Analyze the following customer feedback and provide an analysis for **each feedback entry** in the given list.

        ### Task:
        For each feedback entry, analyze and classify it based on the following:
        1. Sentiment: Positive, Negative, or Neutral.
        2. Category and Sub-category: Select a category and sub-category from the provided list below.
        3. Sentiment Score: A score between 0 and 1, where 1 is most positive and 0 is most negative.

        ### Rules:
        1. Provide exactly one output for each feedback entry. Do not skip any entry.
        2. Maintain the numbering from the input list. The output numbering must match the input numbering.
        3. If you cannot determine the sentiment, category, or sub-category for a feedback entry, return "Unknown" for that field.
        4. Always include a sentiment score, even if the sentiment is "Neutral."
        5. this is important note:-if Thanks, thx , thnkq tnk u , thank u ,Thank YOU!, all kind off thank you in feedback should be positive sentiment and category should be Quick Resolution and sub category as per feedback
        6. if the thank you, thx , thanq all thank you related feedback but if something is more with the thank you then it should go to positive but category and sun category will be by which feedback is with thank you 
        6. dont put ** before the sentiment,category, subcategory etc.
        7. some time rating is different and feedback is different so go with the feedback and put it into correct sentiment and categories and sub categories 
          example like :-
          INC3825539	3	No root cause shared	3	Neutral	General Process Feedback	Non-Specific Comments	0.5
          it shold be in negative but it has rating 3 but check feedback.
        8.dont change category and sub category it should go into corrrect sentiment and category
        9. very important rule:- dont put sub-category in different category
        10. and check rating also rating like 1 means bad and 5 means good it vary in between 1 to 5 so always check rating too.
        11.please make sure that every row must have correct sentement , category, sub category 
        12.dont give the errors
        13.and give exact accurate sentiments,category,sub-category,sentiment score 
        14.sentiment score is on feedback so check the score also whther sentiment will be negative positive or neutral

        ### Categories and Sub-categories:
        - **Positive sentiment**:
          - Quick Resolution category then select subcategory from below only:
            - Quick Support
            - Immediate Resolution
            - Fast Turnaround
            - SLA Adherence
          - Effective Communication category then select subcategory from below only:
            - Clear and Open Communication
            - Frequent Status Updates
            - Transparency in Handling Tickets
            - Acknowledgment with Gratitude
          - Friendly and Supportive Service category then select subcategory from below only:
            - Friendly and Polite Support
            - Respectful and Professional Team
            - Supportive and Understanding Staff
          - Proactive and Personalized Support category then select subcategory from below only:
            - Tailored Solutions
            - Understanding Specific Needs
            - Proactive Problem Identification
            - Preemptive Action
          - Complete and Accurate Resolution category then select subcategory from below only:
            - Comprehensive Fix
            - Accurate Diagnosis
            - Error-Free Resolution
            - Efficient Problem Handling
            - Empathy in Handling Cases
            - Customer-Centric Approach

        - **Neutral sentiment**:
          - General Process Feedback category then select subcategory from below only:
            - Standard Workflow Observations
            - Suggestions for Improvement
          - Ambiguous Feedback category then select subcategory from below only:
            - Non-Specific Comments
            - Neutral Sentiments
            - Generic Input

        - **Negative sentiment**:
          - Delayed Resolution category: then select subcategory from below only
            - Missed Deadlines
            - Slow Ticket Handling
            - Prolonged Resolution Time
            - Repeated Follow-Ups Required
            
          - Incomplete Solution category then select subcategory from below only:
            - Persistent Problems
            - Incomplete Fix
            - Need for Further Investigation
            - Recurring Issues
            - Low Effort in Problem-Solving            
          - Poor Communication category then select subcategory from below only:
            - Lack of Updates
            - Inconsistent Communication
            - Confusing or Misleading Information
            - Acknowledgment Without Resolution
            - Waiting for Response or Updates
            - Acknowledging Delays
          -  Quality Below Expectations category then select subcategory from below only:
            - Dismissive Attitude
            - Lack of Courtesy
            - Disrespectful Interactions
            - Overall Poor Experience
            - Service Quality Below Expectations
            - Hard-to-Follow Procedures
          - No More Support Required category then select subcategory from below only:
            - Resolved Without Support
            - Customer-Fixed Issue
            - Independent Problem Solving
            - Abandoned Support Due to Ineffectiveness
          
            

        ### Feedback to Analyze:
        {combined_feedback}
        the below format should be same every time and how much is input size that much correct size of below output should be
        i am sending a batch size of {batch_size_sending} inputs so below format is also required {batch_size_sending} keep this compolsory
        ### Format (Strictly Follow This Format):
        1. Sentiment: <Positive/Negative/Neutral>, Sentiment Category: <...>, Sentiment Sub-category: <...>, Sentiment Score: <...>
        2. Sentiment: <Positive/Negative/Neutral>, Sentiment Category: <...>, Sentiment Sub-category: <...>, Sentiment Score: <...>
        ...
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-powerbi",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.1
            )

            content = response.choices[0].message.content
            results = []
            for line in content.strip().splitlines():
                parts = line.split(",")
                if len(parts) == 4:
                    sentiment = parts[0].split(":", 1)[1].strip()
                    category = parts[1].split(":", 1)[1].strip()
                    sub_category = parts[2].split(":", 1)[1].strip()
                    sentiment_score = parts[3].split(":", 1)[1].strip()
                    results.append((sentiment, category, sub_category, sentiment_score))
            return results
        except Exception as e:
            st.error(f"Batch OpenAI Error: {e}")
            return [("Error", "Error", "Error", "Error")] * len(feedback_list)

    # Analyze with GenAI
    
    st.write(df)
    if st.button("🚀 Analyze with GenAI (Batch Mode)"):
        with st.spinner("Batch analyzing feedback..."):
            feedbacks = df[selected_column].fillna("").astype(str).tolist()
            # Due = df['Completed with Due'].fillna("").astype(str).tolist()
            # Due = df['Completed with Due'].fillna("").astype(str).tolist()
            
            # ratings = df['u_smiley_survey_rating'].fillna(3).astype(int).tolist()
            st.write(len(feedbacks))
            batch_size = 10
            fedback_size=len(feedbacks)
            more_than_batch_size=fedback_size%10
            number_of_batches=fedback_size//10
            completed_tickets=0
            results = []

            for i in range(0, len(feedbacks), batch_size):
                batch = feedbacks[i:i + batch_size]
                # batch_due=Due[i:i + batch_size]
                # batch_ratings = ratings[i:i + batch_size]
                completed_tickets+=1
                batch_size_sending=10
                if completed_tickets<=number_of_batches:
                    batch_size_sending=10
                else:
                    batch_size_sending=more_than_batch_size
                
                batch_results = batch_analyze(batch,batch_size_sending)
                
                st.write(batch_results)
                results.extend(batch_results)

            # # Ensure results match the DataFrame's length
            # if len(results) < len(df):
            #     missing_rows = len(df) - len(results)
            #     results.extend([("Error", "Error", "Error", "Error")] * missing_rows)

            df[['Sentiment', 'Sentiment Category', 'Sentiment Sub-category', 'Sentiment Score']] = pd.DataFrame(results, index=df.index)
            # st.write("hiii")
            # st.write(df)
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
                    " Quality Below Expectations": ["Dismissive Attitude", "Lack of Courtesy", "Disrespectful Interactions", "Overall Poor Experience", "Service Quality Below Expectations", "Hard-to-Follow Procedures"],
                    "No More Support Required": ["Resolved Without Support", "Customer-Fixed Issue", "Independent Problem Solving", "Abandoned Support Due to Ineffectiveness"]
                }
            }

        # Function to assign the correct category based on sentiment and subcategory
        # def Refresh_DF(row):            
        #     sentiment = row["Sentiment"]
        #     subcategory = row["Sentiment Sub-category"].strip().lower()  # Normalize to lowercase and strip whitespace

        #     if sentiment in sentiment_structure:
        #         for category, subcategories in sentiment_structure[sentiment].items():
        #             normalized_subcategories = [s.lower() for s in subcategories]
        #             # st.write(f"Checking sentiment: {sentiment}, category: {category}, subcategories: {normalized_subcategories}")
        #             # st.write(f"Row subcategory: {subcategory}")
        #             if subcategory in normalized_subcategories:
        #                 return category

        #     st.write(f"No match found for sentiment: {sentiment}, subcategory: {subcategory}")
        #     return None
        st.write(df)
        def update_sentiment_and_category(row):
          subcategory = row["Sentiment Sub-category"].strip().lower()  # Normalize to lowercase and strip whitespace

          for sentiment, categories in sentiment_structure.items():
              for category, subcategories in categories.items():
                  normalized_subcategories = [s.lower() for s in subcategories]
                  if subcategory in normalized_subcategories:
                      return pd.Series({"Sentiment": sentiment, "Sentiment Category": category})

          return pd.Series({"Sentiment": None, "Sentiment Category": None})
        # Apply the function to the DataFrame
        df[["Sentiment", "Sentiment Category"]] = df.apply(update_sentiment_and_category, axis=1)
        df=get_sentiments_by_counts(df)
        # Display the updated DataFrame
        st.write(df)


        st.success("✅ Analysis complete!")
        st.subheader("📄 Processed Data")
        st.session_state.result=df
        st.dataframe(st.session_state.result)

        @st.cache_data
        def convert_df_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            return output.getvalue()

        excel_data = convert_df_to_excel(df)
        st.download_button("📥 Download Processed Excel", data=excel_data, file_name="processed_feedback.xlsx")
        
        