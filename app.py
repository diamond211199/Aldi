import streamlit as st
import pandas as pd
# from anytree import Node, RenderTree
import openai
from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from count import get_count_sentiments
import plotly.express as px
from due import get_sentiments_by_counts
from io import BytesIO
import os
import config
import re
from dotenv import load_dotenv
load_dotenv()
client = openai.AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT")
)


# Set the title in the sidebar
st.sidebar.title("BEAT - Business Emotion Assessment Tracker")

# file_path = 'processed_feedback (5).xlsx'
file_path='Sentiments'
# file_path="https://capgemini.sharepoint.com/sites/upload/Shared Documents/BEAT"
excel_files = [f for f in os.listdir(file_path) if f.endswith('.xlsx') and '~$' not in f]

# Create a dropdown for selecting an Excel file
selected_file = st.sidebar.selectbox("Select an Excel file", excel_files)

# Full path to the selected file
file_path1 = os.path.join(file_path, selected_file)
df = pd.read_excel(file_path1, engine='openpyxl')
df = df.dropna(subset=['Sentiment Category'])
df=df.dropna(subset=['Impacted Service'])
df2=df

option = st.sidebar.radio("Select an option", ["Business KPI Analysis","Sentiment Analysis", "Differencial Analysis","Persona Based Analysis","Inferences","Upload Data"])

# If "Sentiment Analysis" is selected
if option == "Sentiment Analysis":
    
    st.sidebar.title('Impacted Service Analysis')
    # Select unique assignment group
    Impacted_Service = st.sidebar.selectbox('Select Impacted Service', ['All'] + df['Impacted Service'].unique().tolist())
    if Impacted_Service=="All":
        filtered_df1=df
    else:
        filtered_df1 = df[df['Impacted Service'] == Impacted_Service]

    # filtered_df1['u_bpml_2'] = filtered_df1['u_bpml_2'].str.replace(r'^[a-zA-Z0-9]+\.[a-zA-Z0-9]+\.\s*', '', regex=True)
    u_bpml_2 = st.sidebar.selectbox('Select Area',['All']+ filtered_df1['IPO'].unique().tolist())
    if u_bpml_2=="All":
        filtered_df=filtered_df1
    else:
        filtered_df = filtered_df1[filtered_df1['IPO'] == u_bpml_2]


    # Radio buttons for options
    
    sentiment_option = st.sidebar.radio("Select an sentiment", ["Positive", "Neutral", "Negative"])


    sentiment_df = filtered_df[filtered_df['Sentiment'] == sentiment_option]
    # st.write("There are total",str(len(filtered_df))," Tickets Feedback of ",assignment_group," assignment_group")
    min_date=""
    max_date=""
    if 'Opened' in filtered_df.columns:
        # Convert the 'date' column to datetime if it's not already
        df['Opened'] = pd.to_datetime(df['Opened'], errors='coerce')

        # Drop rows with NaT in 'date' column
        df = df.dropna(subset=['Opened'])

        if not df.empty:
            # Find the minimum and maximum dates
            min_date = df['Opened'].min()
            max_date = df['Opened'].max()

            if min_date.year == max_date.year and min_date.month == max_date.month:
                # st.write(f"Data is for {min_date.strftime('%B %Y')}")
                st.markdown(f"<h3>There are total <span style='color: red;'>{len(filtered_df)}</span> Tickets Feedback of {Impacted_Service} assignment group</h3> Data is for <span style='color: red;'>{min_date.strftime('%B %Y')}</span>", unsafe_allow_html=True)

            else:
                # Display the range of months
                # st.write(f"Data from {min_date.strftime('%B %Y')} to {max_date.strftime('%B %Y')}")
                st.markdown(f"<h3>There are total <span style='color: red;'>{len(filtered_df)}</span> Tickets Feedback of {Impacted_Service} assignment group</h3> Data from <span style='color: red;'> {min_date.strftime('%B %Y')} to {max_date.strftime('%B %Y')} </span>", unsafe_allow_html=True)

    
    # .markdown(f"<p style='font-size:13px;'>{pre}{roott.name}</p>", unsafe_allow_html=True)
    sentiment_text="There are "+str(round((len(sentiment_df)/len(filtered_df))*100))+"% ("+str(len(sentiment_df))+") "+sentiment_option+" Ticket Feedback "
    
    st.markdown(f"<h5 style='color: red;'>{sentiment_text}</h5>", unsafe_allow_html=True)
    categories = sentiment_df['Sentiment Category'].unique()
    col1, col2 = st.columns(2)
    len_category=len(categories)
    half_len_category=abs(len_category/2)
    if len_category%2==1:
        half_len_category+=1
    count=1
    for category in categories:
        if count<=half_len_category:
            col=col1
        else:
            col=col2
        print(col)
        # category_text=category+" ("+str(round((len(sentiment_df[sentiment_df['Category'] == category])/len(sentiment_df))*100,1))+"%)"
        category_text = str(category) + " (" + str(round((len(sentiment_df[sentiment_df['Sentiment Category'] == category]) / len(sentiment_df)) * 100, 1)) + "%)"

        col.markdown(f"""<div style="border: 2px solid #d66216; margin-top: 20px; padding: 4px; border-radius: 5px; width: 200px; text-align: center;">
            <strong>{category_text}</strong>
            </div>
            """, unsafe_allow_html=True)

        # root = Node(f"{category} ({round((len(sentiment_df[sentiment_df['Category'] == category])/len(sentiment_df))*100,1)}%)")
        sentiment_text+="from that "+str(category)+" "+str(len(sentiment_df[sentiment_df['Sentiment Category'] == category]))+" category and "
        # column.write(f"- {category} ({len(sentiment_df[sentiment_df['Category'] == category])})")
        sub_df=sentiment_df[sentiment_df['Sentiment Category'] == category]
        subcategories = sub_df['Sentiment Sub-category'].unique()
        for subcategory in subcategories:
            # st.write(f"&nbsp;{subcategory} ({len(sub_df[sub_df['Sub-category'] == subcategory])})")
            
            col.markdown(f"""
            <div style="margin-left: 20px; font-size: 14px;">
            &bull; {subcategory} ({len(sub_df[sub_df['Sentiment Sub-category'] == subcategory])})
            </div>
            """, unsafe_allow_html=True)

            # Node(f"&nbsp;{subcategory} ({len(sub_df[sub_df['Sub-category'] == subcategory])})", parent=root)
            sentiment_text+=str(subcategory)+" "+str(len(sub_df[sub_df['Sentiment Sub-category'] == subcategory]))+" sub category "
            
        count+=1
    if Impacted_Service!='All' and sentiment_option=="Negative":
        count_sentiments_text=get_count_sentiments(file_path1)
        
        def get_selected_negative_sentiments(row):
            return (f"Incident number=\"{row['Number']}\" with Sentiment Sub-category=\"{row['Sentiment Sub-category']}\" ")
            
        
        # st.write(df)
        negative_sentiments_df = filtered_df[filtered_df['Sentiment'] == 'Negative']
        
        row_count=len(negative_sentiments_df)
        # Apply the function to each row
        negative_sentiments_df['sentiments of selected impacted services'] = negative_sentiments_df.apply(get_selected_negative_sentiments, axis=1)
        sentiment_text="total incidents:-"+str(row_count)
        sentiment_text += "\n".join(negative_sentiments_df['sentiments of selected impacted services'].tolist())
        # st.write(sentiment_text)
        prompt_count_sentiments_text = f"""{sentiment_text} these are the negative sentiments of {Impacted_Service} impacted service.
        so check the positive sentiments of other impacted services and give the guidence that how we can leverage others positive sentiments over negative sentiments of impacted service
        Note:- 
        1.just give usefull things, dont give much explaination
        2.give the information of how we can leaverage positive sentiments of other impacted services from below information over negative sentiments of the above information of impacted service  
        3.just give information of how much negative sentiments are there and which are positive sentiments related are there that much only
        below is the all data of all positive negative neutral sentiments of all available impacted services take the positive one:-
        {count_sentiments_text}                                                    
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-powerbi",
                messages=[{"role": "user", "content": prompt_count_sentiments_text}],
                max_tokens=4000,
                temperature=0.1
            )

            # Assuming the response contains a 'choices' list with text in 'message'
            generated_text = response.choices[0].message.content
            print("Model Response:", generated_text)
            # st.write("Model Response:", generated_text)
            st.markdown("<h3 style='color: red;'>Leveraging positive sentiments from other areas</h3>", unsafe_allow_html=True)
            st.write(generated_text)
        except Exception as e:
            print("An error occurred:", e)

    
    
    
    unique_names = sentiment_df['Resolved by'].unique()

    # # Create a dictionary to store names and their unique sub-categories
    # name_subcategories = {}
    st.write("<br>",unsafe_allow_html=True)
    # for name in unique_names:
    #     # Get unique sub-categories for the current name
    #     subcategories = sentiment_df[sentiment_df['Resolved by'] == name]['Sentiment Sub-category'].value_counts()
    #     # Create the text for the current name
        
        
    #     subcategories_text = ", ".join([f"{subcategory} ({count})" for subcategory, count in subcategories.items()])
    #     name_subcategories[name] = f"{name}:- " + subcategories_text
    #     # st.write(subcategories_text)

    # # Print the results
    # # st.write()
    st.markdown("<h3 style='color: red;'>Sentiment analysis w.r.t Resolver</h3>", unsafe_allow_html=True)
    # for name, text in name_subcategories.items():
    #     st.write(text)
        
    
    grouped_df = sentiment_df.groupby(['Resolved by', 'Sentiment Sub-category']).size().reset_index(name='Count')

    # Aggregate the sub-categories and counts into a single string for each person
    aggregated_df = grouped_df.groupby('Resolved by').apply(
        lambda x: ', '.join(f"{row['Sentiment Sub-category']} ({row['Count']})" for _, row in x.iterrows())
    ).reset_index(name='Sentiment Analysis')

    # Create a DataFrame from the collected data
    def create_html_table(df):
        html = """
        <style>
        .styled-table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.9em;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }
        .styled-table thead tr {
            background-color: #009879;
            color: #ffffff;
            text-align: left;
        }
        .styled-table th,
        .styled-table td {
            padding: 12px 15px;
            border: 1px solid #dddddd;
        }
        .styled-table tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        .styled-table tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
        .styled-table tbody tr:last-of-type {
            border-bottom: 2px solid #009879;
        }
        ul {
            padding-left: 20px;
            margin: 0;
        }
        li {
            margin: 0;
        }
        </style>
        <table class="styled-table">
        <thead>
        <tr>"""

        for column in df.columns:
            html += f"<th>{column}</th>"
        html += "</tr></thead><tbody>"

        for _, row in df.iterrows():
            html += "<tr>"
            for column, item in row.items():
                # Apply bullet list transformation only for "Negative" and "Positive" columns
                if column in ["Sentiment Analysis"] and isinstance(item, str) and ',' in item:
                    item = ''.join(f"<li>{str(sub_item).strip()}</li>" for sub_item in item.split(','))
                    item = f"<ul>{item}</ul>"
                html += f"<td>{item}</td>"
            html += "</tr>"

        html += "</tbody></table>"
        return html
    
    st.markdown(create_html_table(aggregated_df), unsafe_allow_html=True)

if option=="Differencial Analysis":
    st.title("Differencial Analysis")
    st.markdown("""
        <p style="font-size: 18px; color: #4CAF50;">
        Compare two categories to analyze their differences in ticket feedback.
        </p>
    """, unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns(2)

    # Dropdown in the first column
    with col1:
        st.session_state.diff1_Region = st.selectbox('Select Impacted Service', ['All']+df['Impacted Service'].unique().tolist(),key="region_1_dropdown")
        if st.session_state.diff1_Region=="All":
            st.session_state.diff1_Region_df1=df
        else:
            st.session_state.diff1_Region_df1 = df[df['Impacted Service'] == st.session_state.diff1_Region]
        # st.session_state.diff1_Region_df1.dropna(subset=['caller_id.u_regions'])
        st.session_state.diff1_assignment_group = st.selectbox('Select Area',['All']+ st.session_state.diff1_Region_df1['IPO'].unique().tolist(),key="category_1_dropdown")
        if st.session_state.diff1_assignment_group=="All":
            st.session_state.diff_filtered_df1 = st.session_state.diff1_Region_df1
        else:
            st.session_state.diff_filtered_df1 = st.session_state.diff1_Region_df1[st.session_state.diff1_Region_df1['IPO'] == st.session_state.diff1_assignment_group]
        st.session_state.diff_filtered_df1['Priority'] = st.session_state.diff_filtered_df1['Priority'].str.replace(r'^[a-zA-Z0-9]+\.[a-zA-Z0-9]+\.\s*', '', regex=True)
        st.session_state.diff1_u_bpml_2 = st.selectbox('Select Priority',['All']+st.session_state.diff_filtered_df1['Priority'].unique().tolist(),key="sub_category_1_dropdown")
        if st.session_state.diff1_u_bpml_2=="All":
            st.session_state.diff_filtered_df1 = st.session_state.diff_filtered_df1
        else:
            st.session_state.diff_filtered_df1 = st.session_state.diff_filtered_df1[st.session_state.diff_filtered_df1['Priority'] == st.session_state.diff1_u_bpml_2]
        st.session_state.diff1_option = st.radio("Select an sentiment", ["Positive", "Neutral", "Negative"],key="sentiment_1_dropdown")
        st.session_state.sentiment_df1 = st.session_state.diff_filtered_df1[st.session_state.diff_filtered_df1['Sentiment'] == st.session_state.diff1_option]
        sentiment_text="There are "+str(round((len(st.session_state.sentiment_df1)/len(st.session_state.diff_filtered_df1))*100))+"% ("+str(len(st.session_state.sentiment_df1))+") "+st.session_state.diff1_option+" Ticket Feedback "
        st.markdown(f"<h5>{sentiment_text}</h5>", unsafe_allow_html=True)
        categories = st.session_state.sentiment_df1['Sentiment Category'].unique()
        
        len_category=len(categories)
        half_len_category=abs(len_category/2)
        if len_category%2==1:
            half_len_category+=1
        count=1
        for category in categories:
            # category_text=category+" ("+str(round((len(sentiment_df[sentiment_df['Category'] == category])/len(sentiment_df))*100,1))+"%)"
            category_text = str(category) + " (" + str(round((len(st.session_state.sentiment_df1[st.session_state.sentiment_df1['Sentiment Category'] == category]) / len(st.session_state.sentiment_df1)) * 100, 1)) + "%)"

            st.markdown(f"""<div style="border: 2px solid #d66216; margin-top: 20px; padding: 4px; border-radius: 5px; width: 200px; text-align: center;">
                <strong>{category_text}</strong>
                </div>
                """, unsafe_allow_html=True)

            # root = Node(f"{category} ({round((len(sentiment_df[sentiment_df['Category'] == category])/len(sentiment_df))*100,1)}%)")
            sentiment_text+="from that "+str(category)+" "+str(len(st.session_state.sentiment_df1[st.session_state.sentiment_df1['Sentiment Category'] == category]))+" category and "
            # column.write(f"- {category} ({len(sentiment_df[sentiment_df['Category'] == category])})")
            sub_df=st.session_state.sentiment_df1[st.session_state.sentiment_df1['Sentiment Category'] == category]
            subcategories = sub_df['Sentiment Sub-category'].unique()
            for subcategory in subcategories:
                # st.write(f"&nbsp;{subcategory} ({len(sub_df[sub_df['Sub-category'] == subcategory])})")
                
                st.markdown(f"""
                <div style="margin-left: 20px; font-size: 14px;">
                &bull; {subcategory} ({len(sub_df[sub_df['Sentiment Sub-category'] == subcategory])})
                </div>
                """, unsafe_allow_html=True)

                # Node(f"&nbsp;{subcategory} ({len(sub_df[sub_df['Sub-category'] == subcategory])})", parent=root)
                sentiment_text+=str(subcategory)+" "+str(len(sub_df[sub_df['Sentiment Sub-category'] == subcategory]))+" sub category "
                
            count+=1
        

    with col2:
        st.session_state.diff2_Region = st.selectbox('Select Impacted Service',['All']+ df['Impacted Service'].unique().tolist(),key="region_2_dropdown")
        if st.session_state.diff2_Region=="All":
            st.session_state.diff2_Region_df2=df
        else:
            st.session_state.diff2_Region_df2 = df[df['Impacted Service'] == st.session_state.diff2_Region]
        # st.session_state.diff2_Region_df2 = df[df['caller_id.u_regions'] == st.session_state.diff2_Region]
        st.session_state.diff2_assignment_group = st.selectbox('Select Area', ['All']+st.session_state.diff2_Region_df2 ['IPO'].unique().tolist(),key="category_2_dropdown")
        if st.session_state.diff2_assignment_group=="All":
            st.session_state.diff_filtered_df2=st.session_state.diff2_Region_df2
        else:
            st.session_state.diff_filtered_df2 = st.session_state.diff2_Region_df2[st.session_state.diff2_Region_df2['IPO'] == st.session_state.diff2_assignment_group]
        st.session_state.diff_filtered_df2['Priority'] = st.session_state.diff_filtered_df2['Priority'].str.replace(r'^[a-zA-Z0-9]+\.[a-zA-Z0-9]+\.\s*', '', regex=True)
        st.session_state.diff2_u_bpml_2 = st.selectbox('Select Priority',['All']+st.session_state.diff_filtered_df2['Priority'].unique().tolist(),key="sub_category_2_dropdown")
        if st.session_state.diff2_u_bpml_2=="All":
            st.session_state.diff_filtered_df2=st.session_state.diff_filtered_df2
        else:
            st.session_state.diff_filtered_df2 = st.session_state.diff_filtered_df2[st.session_state.diff_filtered_df2['Priority'] == st.session_state.diff2_u_bpml_2]
       
        st.session_state.diff2_option = st.radio("Select an sentiment", ["Positive", "Neutral", "Negative"],key="sentiment_2_dropdown")
        st.session_state.sentiment_df2 = st.session_state.diff_filtered_df2[st.session_state.diff_filtered_df2['Sentiment'] == st.session_state.diff2_option]
        sentiment_text="There are "+str(round((len(st.session_state.sentiment_df2)/len(st.session_state.diff_filtered_df2))*100))+"% ("+str(len(st.session_state.sentiment_df2))+") "+st.session_state.diff2_option+" Ticket Feedback "
        st.markdown(f"<h5>{sentiment_text}</h5>", unsafe_allow_html=True)
        categories = st.session_state.sentiment_df2['Sentiment Category'].unique()
        
        len_category=len(categories)
        half_len_category=abs(len_category/2)
        if len_category%2==1:
            half_len_category+=1
        count=1
        for category in categories:
            # category_text=category+" ("+str(round((len(sentiment_df[sentiment_df['Category'] == category])/len(sentiment_df))*100,1))+"%)"
            category_text = str(category) + " (" + str(round((len(st.session_state.sentiment_df2[st.session_state.sentiment_df2['Sentiment Category'] == category]) / len(st.session_state.sentiment_df2)) * 100, 1)) + "%)"

            st.markdown(f"""<div style="border: 2px solid #d66216; margin-top: 20px; padding: 4px; border-radius: 5px; width: 200px; text-align: center;">
                <strong>{category_text}</strong>
                </div>
                """, unsafe_allow_html=True)

            # root = Node(f"{category} ({round((len(sentiment_df[sentiment_df['Category'] == category])/len(sentiment_df))*100,1)}%)")
            sentiment_text+="from that "+str(category)+" "+str(len(st.session_state.sentiment_df2[st.session_state.sentiment_df2['Sentiment Category'] == category]))+" category and "
            # column.write(f"- {category} ({len(sentiment_df[sentiment_df['Category'] == category])})")
            sub_df=st.session_state.sentiment_df2[st.session_state.sentiment_df2['Sentiment Category'] == category]
            subcategories = sub_df['Sentiment Sub-category'].unique()
            for subcategory in subcategories:
                # st.write(f"&nbsp;{subcategory} ({len(sub_df[sub_df['Sub-category'] == subcategory])})")
                
                st.markdown(f"""
                <div style="margin-left: 20px; font-size: 14px;">
                &bull; {subcategory} ({len(sub_df[sub_df['Sentiment Sub-category'] == subcategory])})
                </div>
                """, unsafe_allow_html=True)

                # Node(f"&nbsp;{subcategory} ({len(sub_df[sub_df['Sub-category'] == subcategory])})", parent=root)
                sentiment_text+=str(subcategory)+" "+str(len(sub_df[sub_df['Sentiment Sub-category'] == subcategory]))+" sub category "
                
            count+=1
            
            
    st.write("<br>",unsafe_allow_html=True)
    # st.markdown("<h3 style='color: red;'></h3>", unsafe_allow_html=True)
            
                
if option=="Business KPI Analysis":
    
    unique_services = df['Impacted Service'].unique().tolist()

    # Set the first value from the list as the default
    default_value = unique_services[0]

    # Create the selectbox with the default value
    Impacted_Service = st.sidebar.selectbox(
        'Select Impacted Service', ['All'] + unique_services, index=(['All'] + unique_services).index(default_value)
    )
    if Impacted_Service=="All":
        filtered_df1=df
    else:
        filtered_df1 = df[df['Impacted Service'] == Impacted_Service]

    # filtered_df1['u_bpml_2'] = filtered_df1['u_bpml_2'].str.replace(r'^[a-zA-Z0-9]+\.[a-zA-Z0-9]+\.\s*', '', regex=True)
    Selected_IPO = st.sidebar.selectbox('Select Area',['All']+ filtered_df1['IPO'].unique().tolist())
    if Selected_IPO=="All":
        filtered_df=filtered_df1
    else:
        filtered_df = filtered_df1[filtered_df1['IPO'] == Selected_IPO]
        
    Selected_Priority = st.sidebar.selectbox('Select Priority',['All']+ filtered_df['Priority'].unique().tolist())
    if Selected_Priority=="All":
        filtered_df2=filtered_df
    else:
        filtered_df2 = filtered_df[filtered_df['Priority'] == Selected_Priority]
      
    filtered_df3=filtered_df2  
    filtered_df3['Resolved'] = pd.to_datetime(filtered_df3['Resolved'])

    # Extract unique month-year pairs
    filtered_df3['Month-Year'] = filtered_df3['Resolved'].dt.strftime('%B %Y')
    unique_month_years = filtered_df3['Month-Year'].dropna().unique().tolist()

    # Set the first value from the list as the default
    default_value_month = unique_month_years[0]

    # Selected_Month_Year = st.sidebar.selectbox('Select Month-Year', ['All'] + unique_month_years)
    Selected_Month_Year = st.sidebar.selectbox(
        'Select Month-Year', ['All'] + unique_month_years, index=(['All'] + unique_month_years).index(default_value_month)
    )
    # Filter DataFrame based on the selected month-year
    if Selected_Month_Year == "All":
        filtered_df4 = filtered_df2
        # st.write("Displaying data for all months.")
    else:
        filtered_df4 = filtered_df2[filtered_df2['Month-Year'] == Selected_Month_Year]
        # st.write(f"Displaying data for {Selected_Month_Year}.")

    # Display the filtered DataFrame
    # st.write(filtered_df4)
    
    # st.write(filtered_df2)
    
    st.markdown("<h3 style='color: red;'>Mean Time To Resolve Incidents</h3>", unsafe_allow_html=True)
            
    df=filtered_df4
    row_count = len(filtered_df4)
    average_time_period = df['Days To Complete'].mean()

    # Plotly figure
    fig = go.Figure()

    # Add line and markers
    fig.add_trace(go.Scatter(x=df['Number'], y=df['Days To Complete'],
                            mode='lines+markers', name='Time Period'))

    # Add average line
    fig.add_trace(go.Scatter(x=df['Number'], y=[average_time_period] * len(df),
                            mode='lines', name='Average', line=dict(dash='dash', width=4, color='red')))

    # Update layout
    fig.update_layout(title='',
                    xaxis_title='Incidents',
                    yaxis_title='Time Period (Days)')

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    
    def resolve_message(row):
        if row['Days Late'] < 0:
            return (f"Incident number=\"{row['Number']}\" with description=\"{row['Description']}\" with close notes=\"{row['Close notes/Resolution Notes']}\" "
                    f"has resolved in {row['Days To Complete']} days and it has completed "
                    f"{-row['Days Late']} days before expected time.")
        else:
            return (f"Incident number=\"{row['Number']}\" with description=\"{row['Description']}\" with close notes=\"{row['Close notes/Resolution Notes']}\" "
                    f"has resolved in {row['Days To Complete']} days and it has completed "
                    f"{row['Days Late']} days late.")
    
    # st.write(df)
    
    # Apply the function to each row
    df['Resolution Message'] = df.apply(resolve_message, axis=1)
    resolution_text="total incidents:-"+str(row_count)
    resolution_text += "\n".join(df['Resolution Message'].tolist())
    
   
    prompt = f"""
        Given data for a Aldi retailer indicates that impacted service:- {Impacted_Service} is impacted.
        1. Identify the Business KPI(s) that are likely to be impacted by these incidents.
            Only provide 3 Kpi's 
            for some of the areas we have KPI(s) mentioning below consider those also if they are impacted as per above mentioned impacted service other wise give as per your knowledge and give information as well:-
            note:- use it if and if only it is impacted and as per impacted service mentioned above dont go for all below mentioned and if not impacting as per below data
            impacted service:-Order to Cash:
                On time In full %
                Order to delivery cycle time
                
            impacted service:-Procure to Pay:
                On time In full % (GR)
                On time invoice payment %
                Procure to Pay cycle time
                
            impacted service:-WareHouse managment:
                Pick to Pack cycle time
                On time shipping %
                %Picking accuracy
                
            impacted service:-Inventory managment:
                % inventory availability
                % inventory accuracy  
                  
            impacted service:-Contract managment:
                Contract ageing report
                No. of invoices due for expire
                
            impacted service:-Transportation managment:
                Ship to Delivery cycle time
                Truck turnaround time

        2. Quantify the impact on these KPIs, providing detailed information and estimating the impact percentage due to the delayed incidents.

        3. Identify the root cause(s) of the incidents and provide the count for each cause dont provide incident numbers and give it in sentence.And give count of each root cause repeted
        4. we have BIS Automation Enablers that we can use it for our root causes or problems which impacting our KPI's or Business Area's the list as given below from that list which automation enabler will be helpful for the above given impacted KPI's:-
            1.Auto Release Of Sales Order Blocked Due To Credit Hold
            2.Automation of SRM Organization Hierarchy Extraction
            3.Automation of Store Closure Process
            4.Automation to manage bulk PO cancellation process
            5.Bulk Creation and Updation of SKUs
            6.Check Duplicate Invoice And Trigger Cancellation (DSO)
            7.Credit Check Failure Notification
            8.Daily Purchase Order (PO) Interface failure
            9.Data Validation Program
            10.Delays in month-end closing
            11.Duplicate Purchase Order
            12.G/L Reconciliation
            13.Handling Orphan Payments and Invoices
            14.Health of Ecommerce website
            15.High value PO getting stuck due to TMS
            16.Inventory Accuracy
            17.Loyalty Program Management
            18.Manage Auto Release of Orders blocked due to Credit Hold
            19.Manage Clarifications for Payments
            20.Manage Customer Retention
            21.Manage Data Inconsistencies
            22.Manage delayed shopping cart Processing
            23.Manage Excess Stock and Purchase Price Difference in a Plant
            24.Manage Inconsistency in supplier information
            25.Manage Inconsistent Information in Procurement Process
            26.Manage Inventory accuracy
            27.Manage Invoice Processing Exception
            28.Manage Invoice Quantity Mismatch Discrepancies
            29.Manage Orphan Payments And Invoices (DSO)
            30.Manage Purchase Request Exceptions
            31.Mass PO_SO Upload
            32.Order creation failure notification & fix
            33.Planogram - Accurate Presentation Management
            34.PO Interface to Warehouse
            35.PO Job Failure Alerts to Supplier
            36.PO Release Tracking and Auto Scheduling
            37.PR approval hierarchy
            38.PRA Journal Entry Adjustment
            39.Preventing Duplicate Orders
            40.Price Accuracy
            41.Price and Promo Accuracy
            42.Proactive Alert for Delays in Delivery of PO Items
            43.Scan & Go
            44.Shipment Notification Failure
            45.Standardization of Discount Quotes
            46.STO(Stock Transfer Order) monitoring dashboard to represent STO RAG status
            47.Manage invalid and missing serial numbers for revenue realization in Order to ship business process
            48.Manage PO sync issue
            49.Manage special character in invoice
            50.Shipment Monitor (Double Shipments)
            51.Maintain price validity through data sync between ECC and SAP Sourcing system
            52.BNDC: Configure self heal solutions for IDOCs in error
            53.SNB: Predict stock changes in procurement and sales
             
        5. Provide just percentage and due to in Smart AM KPI for the given Below Data. it is like the issue impacting business process due to and how much that is mean by Smart AM KPI
           show it like in list as per below example use % symbole dont give percentage value
           1. % activity delayed due to root cause and reason if any increase decreased
           2. % impacted KPI delay due to root cause and reason if any increase decreased
           .
           .
            
           for example:-
           1. % Pick to Pack delayed due to incorrect weight verification
           .
           .

        **imp:- must follow below display dont give any other information
        how to display it:-show the all things like below example and this all should be headings:-
            like suppose we are impacting 3 kpis then show it like
            1st impacted kpi
            quanitify impact
             >
             >
            root cause
             >
             >
            smart AM KPI
             >
             >
            automation enablers usefull
             >
             >
            2nd impacted kpi
            quanitify impact
             >
             >
            root cause
             >
             >
            smart AM KPI
             >
             >
            automation enablers usefull
             >
             >
        same for all
        Note: Please provide a concise and short response , following the instructions, with just a few sentences.just give the information dont show other stuff like formulas ETC.

        {resolution_text}
        """
    if st.button("Business KPI(s) impacted"):        
        if Impacted_Service!='All':
            # st.write("hii")
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-powerbi",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.1
                )

                # Assuming the response contains a 'choices' list with text in 'message'
                generated_text = response.choices[0].message.content
                print("Model Response:", generated_text)
                # st.write("Model Response:", generated_text)
                st.markdown("<h3 style='color: red;'>Business KPI Impacted</h3>", unsafe_allow_html=True)
                st.write(generated_text)
            except Exception as e:
                print("An error occurred:", e)
                st.write("An error occurred:", e)
            
      
    
    avg_days1 = df.groupby('Impacted Service')['Days To Complete'].mean().reset_index()
    
    st.markdown("<h3 style='color: red;'>Average Days To Complete Incidents by Impacted Service</h3>", unsafe_allow_html=True)
            
    num_services = avg_days1['Impacted Service'].nunique()

    # Set rotation angle based on the number of unique services
    tick_angle = 0 if num_services <= 5 else -45

    # Create an interactive bar chart with Plotly
    fig = px.bar(avg_days1, 
                 x='Impacted Service', 
                 y='Days To Complete',
                 text='Days To Complete',
                 labels={'Days To Complete': 'Average Days To Complete Incidents'},
                 title='',
                 hover_data={'Days To Complete': ':.2f'})  # Format hover data to 2 decimal places

    # Customize the layout for better readability
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', marker_line_width=1.5)
    fig.update_layout(xaxis_title='Impacted Service',
                      yaxis_title='Average Days To Complete Incidents',
                      xaxis_tickangle=tick_angle,  # Rotate x-axis labels based on the number of categories
                      margin=dict(l=40, r=40, t=40, b=100))

    # Set a minimum bar width
    bar_width = 0.2 if num_services == 1 else None
    fig.update_traces(width=bar_width)

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    
    
    
    
    avg_days = df.groupby('Priority')['Days To Complete'].mean().reset_index()

    # st.write(avg_days)
    df['Is Delayed'] = df['Days Late'] > 0

    # Create a summary DataFrame with information about delayed incidents for each priority
    # delayed_info = df[df['Is Delayed']].groupby('Priority').apply(
    #     lambda x: ', '.join(x['Number'].astype(str))
    # ).reset_index(name='Delayed Incidents')
    
    df['Is Delayed'] = df['Days Late'] > 0

    # Group by priority and calculate delayed incidents and their count
    delayed_info = df[df['Is Delayed']].groupby('Priority').apply(
        lambda x: {
            'Delayed Incidents': ', '.join(x['Number'].astype(str)),
            'Count of Delayed Incidents': x['Number'].count()
        }
        ).reindex(fill_value={'Delayed Incidents': '', 'Count of Delayed Incidents': 0}).reset_index()

    # If a priority has no delayed incidents, ensure it's represented with a count of 0
    delayed_info = delayed_info.fillna({'Delayed Incidents': '', 'Count of Delayed Incidents': 0})

    # Adjust the DataFrame to have separate columns for clarity
    delayed_info = delayed_info.join(delayed_info.pop(0).apply(pd.Series))

    # Print the results
    print(delayed_info)
    
    avg_days = avg_days.merge(delayed_info, on='Priority', how='left')
        
    st.markdown("<h3 style='color: red;'>Average Days To Complete Incidents by Priority</h3>", unsafe_allow_html=True)
            
    num_services = avg_days['Priority'].nunique()

    # Set rotation angle based on the number of unique services
    tick_angle = 0 if num_services <= 5 else -45

    # Create an interactive bar chart with Plotly
    fig = px.bar(avg_days, 
                 x='Priority', 
                 y='Days To Complete',
                 text='Days To Complete',
                 labels={'Days To Complete': 'Average Days To Complete Incident'},
                 title='',
                 hover_data={'Days To Complete': ':.2f','Delayed Incidents': True })  # Format hover data to 2 decimal places

    # Customize the layout for better readability
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', marker_line_width=1.5)
    fig.update_layout(xaxis_title='Priority',
                      yaxis_title='Average Days To Complete Incidents',
                      xaxis_tickangle=tick_angle,  # Rotate x-axis labels based on the number of categories
                      margin=dict(l=40, r=40, t=40, b=100))

    # Set a minimum bar width
    bar_width = 0.2 if num_services == 1 else None
    fig.update_traces(width=bar_width)

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    
if option=="Persona Based Analysis":
    
    
    def extract_name(full_name):
        # Remove email part and any content in parentheses
        return re.sub(r'\s*\(.*?\)', '', full_name).strip()

    # Check if 'Role' column exists
    if 'Role' not in df.columns:
        # Add a new column for plain names
        df['Plain Name'] = df['Opened by'].apply(extract_name)

        # # Grouping and preparing data
        # agg_df = df.groupby(['IPO', 'Plain Name', 'Sentiment'])['Sentiment Sub-category'].apply(list).reset_index()

        unique_names = df['Plain Name'].unique()
        # st.write(len(unique_names))

        # Construct detailed info for prompt
        detailed_info = []
        for name in unique_names:
            person_data = df[df['Plain Name'] == name]
            Short_description = person_data['Short description'].iloc[0]
            Assignment_group = person_data['Assignment group'].iloc[0]
            Tower = person_data['Tower'].iloc[0]
            Application = person_data['Application'].iloc[0]
            Impacted_Service = person_data['Impacted Service'].iloc[0]
            detailed_info.append(f"name: {name} (Short description: {Short_description}, Assignment group: {Assignment_group}, Tower: {Tower}, Application: {Application}, Impacted Service: {Impacted_Service})")

        names_string = "; ".join(detailed_info)
        # st.write(agg_df)

        prompt = f"From the data provided of Aldi company, provide Persona of each contact who has raised the incident: {names_string}. Provide the result  dont give any more information just give data  in the format 'Name: Role'."

        try:
            response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.1
            )

            response_text = response.choices[0].message.content
            st.write(response_text)

            name_role_dict = {}
            for line in response_text.split('\n'):
                if ':' in line:
                    try:
                        number_name, role = line.split(':', 1)
                        _, name = number_name.split('.', 1)
                        name_role_dict[name.strip()] = role.strip()
                    except ValueError as ve:
                        st.error(f"Error parsing line: {line}. Details: {ve}")

            # st.write(name_role_dict)
            for i, name in enumerate(df['Plain Name']):
                # Check if the name is in the name_role_dict
                if name in name_role_dict:
                    # Assign the role to the corresponding row in the DataFrame
                    df.at[i, 'Role'] = name_role_dict[name]
           
            df.to_excel(file_path1, index=False)
            st.success(f"Excel file '{file_path1}' updated successfully!")

        except Exception as e:
            st.error(f"Batch OpenAI Error: {e}")
    else:
        print("Role column already exists in the DataFrame.")
        # st.write(df)
    
    
    
    # st.write(filtered_df)
    def get_unique_with_counts(sub_category_list):
        sub_category_series = pd.Series(sub_category_list)
        counts = sub_category_series.value_counts()
        return [f"{sub} ({count})" for sub, count in counts.items()]

    st.markdown("<h3 style='color: red;'>Persona Based Analysis</h3>", unsafe_allow_html=True)
      
    
    

    # Ensure we have columns for both positive and negative sentiments
    selected_sentiment = st.selectbox("Select Sentiment",["Negative","Positive"])
    # if selected_sentiment!="All":
    df=df[df['Sentiment']==selected_sentiment]
        
    roles=df['Role'].dropna().unique().tolist()
    
    selected_role = st.selectbox("Select Role",["All"]+roles)
    if selected_role!="All":
        df=df[df['Role']==selected_role]

    filtered_df = df[df['Sentiment'].isin(['Positive', 'Negative'])]
    agg_df = filtered_df.groupby(['Role','Plain Name', 'Sentiment'])['Sentiment Sub-category'].apply(list).unstack(fill_value=[])
    agg_df = agg_df.reset_index()
    
    if 'Positive' not in agg_df.columns and selected_sentiment=="Positive":
        agg_df['Positive'] = []
    if 'Negative' not in agg_df.columns and selected_sentiment=="Negative":
        agg_df['Negative'] = []

    # Convert lists to strings for better readability in the DataFrame output
    if selected_sentiment=="Positive":
        agg_df['Positive'] = agg_df['Positive'].apply(lambda x: ', '.join(get_unique_with_counts(x)))
    if selected_sentiment=="Negative":
        agg_df['Negative'] = agg_df['Negative'].apply(lambda x: ', '.join(get_unique_with_counts(x)))
    # st.write(agg_df)
    
    agg_df.columns.values[0] = 'Persona'
    agg_df.columns.values[1] = 'Person' 
    # Display the table using HTML and CSS for better formatting
    
    def create_html_table(df):
        html = """
        <style>
        .styled-table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.9em;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }
        .styled-table thead tr {
            background-color: #009879;
            color: #ffffff;
            text-align: left;
        }
        .styled-table th,
        .styled-table td {
            padding: 12px 15px;
            border: 1px solid #dddddd;
        }
        .styled-table tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        .styled-table tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
        .styled-table tbody tr:last-of-type {
            border-bottom: 2px solid #009879;
        }
        ul {
            padding-left: 20px;
            margin: 0;
        }
        li {
            margin: 0;
        }
        </style>
        <table class="styled-table">
        <thead>
        <tr>"""

        for column in df.columns:
            html += f"<th>{column}</th>"
        html += "</tr></thead><tbody>"

        for _, row in df.iterrows():
            html += "<tr>"
            for column, item in row.items():
                # Apply bullet list transformation only for "Negative" and "Positive" columns
                if column in ["Negative", "Positive"] and isinstance(item, str) and ',' in item:
                    item = ''.join(f"<li>{str(sub_item).strip()}</li>" for sub_item in item.split(','))
                    item = f"<ul>{item}</ul>"
                html += f"<td>{item}</td>"
            html += "</tr>"

        html += "</tbody></table>"
        return html


    # # Using st.markdown to display the HTML table
    st.markdown(create_html_table(agg_df), unsafe_allow_html=True)

    # st.write("<br>",unsafe_allow_html=True)
    

    # # Group by company and sentiments, then aggregate sub-categories into lists
    # agg_df = filtered_df.groupby(['Assignment group', 'Sentiment'])['Sentiment Sub-category'].apply(list).unstack(fill_value=[])

    # # Reset index to make it a proper DataFrame
    # agg_df = agg_df.reset_index()

    # # Ensure we have columns for both positive and negative sentiments
    # if 'Positive' not in agg_df.columns:
    #     agg_df['Positive'] = []
    # if 'Negative' not in agg_df.columns:
    #     agg_df['Negative'] = []

    # # Convert lists to strings for better readability in the DataFrame output
    # agg_df['Positive'] = agg_df['Positive'].apply(lambda x: ', '.join(get_unique_with_counts(x)))
    # agg_df['Negative'] = agg_df['Negative'].apply(lambda x: ', '.join(get_unique_with_counts(x)))

    
    # agg_df.columns.values[0] = 'Business Process' 
    # # Display the table using HTML and CSS for better formatting
    # def create_html_table(df):
    #     html = """
    #     <style>
    #     .styled-table {
    #         width: 100%;
    #         border-collapse: collapse;
    #         margin: 25px 0;
    #         font-size: 0.9em;
    #         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    #         min-width: 400px;
    #         box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    #     }
    #     .styled-table thead tr {
    #         background-color: #009879;
    #         color: #ffffff;
    #         text-align: left;
    #     }
    #     .styled-table th,
    #     .styled-table td {
    #         padding: 12px 15px;
    #         border: 1px solid #dddddd;
    #     }
    #     .styled-table tbody tr {
    #         border-bottom: 1px solid #dddddd;
    #     }
    #     .styled-table tbody tr:nth-of-type(even) {
    #         background-color: #f3f3f3;
    #     }
    #     .styled-table tbody tr:last-of-type {
    #         border-bottom: 2px solid #009879;
    #     }
    #     ul {
    #         padding-left: 20px;
    #         margin: 0;
    #     }
    #     li {
    #         margin: 0;
    #     }
    #     </style>
    #     <table class="styled-table">
    #     <thead>
    #     <tr>"""

    #     for column in df.columns:
    #         html += f"<th>{column}</th>"
    #     html += "</tr></thead><tbody>"

    #     for _, row in df.iterrows():
    #         html += "<tr>"
    #         for column, item in row.items():
    #             # Apply bullet list transformation only for "Negative" and "Positive" columns
    #             if column in ["Negative", "Positive"] and isinstance(item, str) and ',' in item:
    #                 item = ''.join(f"<li>{str(sub_item).strip()}</li>" for sub_item in item.split(','))
    #                 item = f"<ul>{item}</ul>"
    #             html += f"<td>{item}</td>"
    #         html += "</tr>"

    #     html += "</tbody></table>"
    #     return html


    # Using st.markdown to display the HTML table
    # st.markdown(create_html_table(agg_df), unsafe_allow_html=True)

if option=="Inferences":
    
    
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-powerbi",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.1
        )

        # Assuming the response contains a 'choices' list with text in 'message'
        generated_text = response.choices[0].message.content
        print("Model Response:", generated_text)
        # st.write("Model Response:", generated_text)
        st.markdown("<h3 style='color: red;'>Business KPI Impacted</h3>", unsafe_allow_html=True)
        st.write(generated_text)
    except Exception as e:
        print("An error occurred:", e)

    st.write("Top 3 Impacted Services By Ticket Counts")
    impacted_service_counts = df['Impacted Service'].value_counts()

    # Get the top 3 services with the most negative sentiments
    top_3_negative_services = impacted_service_counts.head(3)
    
    st.write(top_3_negative_services)
    
    st.write("Top 3 Impacted Services By Negative Sentiments")
    negative_df = df[df['Sentiment'] == 'Negative']

    # Count negative occurrences for each service
    service_negative_counts = negative_df['Impacted Service'].value_counts()

    # Get the top 3 services with the most negative sentiments
    top_3_negative_services = service_negative_counts.head(3)
    # st.write(top_3_negative_services.columns)
    st.write(top_3_negative_services)
    
    
    
    total=len(df)
    Negative=(len(df[df['Sentiment']=='Negative'])/total)*100
    Positive=(len(df[df['Sentiment']=='Positive'])/total)*100
    Neutral=(len(df[df['Sentiment']=='Neutral'])/total)*100
    categories=['Negative','Positive',"Neutral"]
    values=[Negative,Positive,Neutral]
    # Create a pie chart using Plotly
    fig = px.pie(
        names=categories, 
        values=values, 
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    # Update layout for a smaller chart
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=8)
    fig.update_layout(
        width=350,  # Width of the plot
        height=350,  # Height of the plot
        margin=dict(t=20, b=20, l=20, r=20),  # Adjust margins
    )

    # Streamlit app
    st.title('Overall Sentiments')

    st.plotly_chart(fig, use_container_width=False)



if option == "Upload Data":
    st.title("🔍 BEAT - Business Emotion Assessment Tracker")

    uploaded_file = st.file_uploader("📤 Upload an Excel file", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df = df.head(1000)  # Limit to first 100 rows for performance
        
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
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6
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
        
        # st.write(df)
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
                    
                    # st.write(batch_results)
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
            # st.write(df)


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
            

            # Automatically save the DataFrame to an Excel file in the specified directory
            os.makedirs(config.directory, exist_ok=True)
            file_name="Processed_Data.xlsx"
            # Construct the full file path
            file_path = os.path.join(config.directory, file_name)

            # Use pandas to write the DataFrame to an Excel file
            df.to_excel(file_path, index=False)

            # Inform the user that the file has been saved
            st.write(f"Data has been automatically saved to {file_path}")
            excel_data = convert_df_to_excel(df)
            st.download_button("📥 Download Processed Excel", data=excel_data, file_name="processed_feedback.xlsx")
            
            
        