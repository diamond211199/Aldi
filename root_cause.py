import streamlit as st
import pandas as pd
import openai
from io import BytesIO
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

def batch_analyze(feedback_list, batch_size_sending):
    # Combine feedbacks into a single string for the prompt
    combined_feedback = "\n".join(
        [f"{i+1}. Feedback: {feedback_list[i].strip().lower()}" for i in range(len(feedback_list))]
    )
    prompt = f"""
    you are a very intelligent business analyst. You have to do ticket analysis and find the 1.root cause,2.what issue, 3.where is the issue of the tickets from the description provided.
    break down the incidents more meaningfully by analyzing the context and identifying 
    the actual issues based on the descriptions and resolution notes. 
    Here’s a more thoughtful categorization and sub categorization of it along with root cause 
    from these ticket data, make it more customize as in what was the issue in Bucket,
    and why was that issue in sub bucket, and a root cause of why,
    don't add-  fic and fix date and next steps ?, in issue, don't copy paste items, 
    do more logical reasoning and understand and search the meaning of it.
    it would be very helpful if you add a due to scenario in sub bucket as in,
    invoice and PO visibility issue due to whatever the reason is
    ### Feedback to Analyze:
    {combined_feedback}

    The below format should be the same every time, and the output size should match the input size dont provide any more heading or different things just below structure .
    I'm sending a batch size of {batch_size_sending} inputs, so the format is also required for {batch_size_sending}.so give me the {batch_size_sending} results in below format and dont give comma(,) in result only use in between root cause , where , what Keep this compulsory.
    ### Format (Strictly Follow This Format):
    1. root cause: <causes>, what: <what issue>, where : <where issue>
    2. root cause: <causes>, what: <what issue>, where : <where issue>
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
            if len(parts) == 3:
                root_causes = parts[0].split(":", 1)[1].strip()
                what = parts[1].split(":", 1)[1].strip()
                where = parts[2].split(":", 1)[1].strip()
                results.append((root_causes, what, where))
        # If parsing fails, fill with error
        if len(results) != batch_size_sending:
            results = [("Error", "Error", "Error")] * batch_size_sending
        return results
    except Exception as e:
        st.error(f"Batch OpenAI Error: {e}")
        return [("Error", "Error", "Error")] * batch_size_sending

@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = df.head(27)

    st.subheader("📌 Select Feedback Columns")
    selected_columns = st.multiselect(
        "Choose the columns you want to include (e.g., feedback, short description, resolution note):",
        df.columns
    )

    if selected_columns:
        # Drop rows with missing values in any selected column
        df = df.dropna(subset=selected_columns).reset_index(drop=True)
        st.write("Selected Data Preview:", df[selected_columns].head())

        # Prepare empty columns for results
        df['root_causes'] = None
        df['what'] = None
        df['where'] = None

        if st.button("🚀 Analyze with GenAI (Batch Mode)"):
            with st.spinner("Batch analyzing feedback..."):
                # Combine selected columns into a single string per row
                feedbacks = df[selected_columns].astype(str).agg(" | ".join, axis=1).tolist()
                batch_size = 10
                total = len(feedbacks)
                progress = st.progress(0)
                for i in range(0, total, batch_size):
                    batch = feedbacks[i:i + batch_size]
                    batch_size_sending = len(batch)
                    batch_results = batch_analyze(batch, batch_size_sending)
                    # Update the DataFrame in-place for this batch
                    for j, (root, what, where) in enumerate(batch_results):
                        df.at[i + j, 'root_causes'] = root
                        df.at[i + j, 'what'] = what
                        df.at[i + j, 'where'] = where
                    # Update progress bar
                    progress.progress(min((i + batch_size) / total, 1.0))

                st.success("✅ Analysis complete!")
                st.subheader("📄 Processed Data")
                st.session_state.result = df
                st.dataframe(st.session_state.result)

                excel_data = convert_df_to_excel(df)
                st.download_button("📥 Download Processed Excel", data=excel_data, file_name="processed_feedback.xlsx")
    else:
        st.warning("Please select at least one column.")