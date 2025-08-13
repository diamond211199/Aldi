
    # # Construct detailed info for prompt
    # detailed_info = []
    # for name in unique_names:
    #     person_data = filtered_df[filtered_df['Plain Name'] == name]
    #     Short_description = person_data['Short description'].iloc[0]
    #     Assignment_group = person_data['Assignment group'].iloc[0]
    #     Tower = person_data['Tower'].iloc[0]
    #     Application = person_data['Application'].iloc[0]
    #     Impacted_Service = person_data['Impacted Service'].iloc[0]
    #     detailed_info.append(f"name: {name} (Short description: {Short_description}, Assignment group: {Assignment_group}, Tower: {Tower}, Application: {Application}, Impacted Service: {Impacted_Service})")

    # names_string = "; ".join(detailed_info)
    # st.write(agg_df)

    # prompt = f"From the data provided of Aldi company, provide Persona of each contact who has raised the incident: {names_string}. Provide the result  dont give any more information just give data  in the format 'Name: Role'."

    # try:
    #     response = client.chat.completions.create(
    #         model="gpt-4o-powerbi",
    #         messages=[{"role": "user", "content": prompt}],
    #         max_tokens=4000,
    #         temperature=0.1
    #     )

    #     response_text = response.choices[0].message.content
    #     st.write(response_text)

    #     # Parse the response text to create name-role dictionary
    #     name_role_dict = {}
    #     for line in response_text.strip().split('\n'):
    #         if ':' in line:
    #             number_name, role = line.split(':', 1)
    #             # Remove numbering and strip whitespace
    #             _, name = number_name.split('.', 1)
    #             name_role_dict[name.strip()] = role.strip()

    #     # Convert the name_role_dict to a DataFrame
    #     roles_df = pd.DataFrame(list(name_role_dict.items()), columns=['Plain Name', 'Role'])

    #     st.write("Roles DataFrame:")
    #     st.write(roles_df)

    #     # Manual assignment of roles
    #     agg_df['Role'] = agg_df.apply(lambda row: name_role_dict.get(row['Plain Name'], 'Unknown'), axis=1)

    #     # Debugging: Check the output of the role mapping
    #     st.write("Updated agg_df with Roles:")
    #     st.write(agg_df)

    # except Exception as e:
    #     st.error(f"Batch OpenAI Error: {e}")
    
    
    # # Reset index to make it a proper DataFrame
    # agg_df = agg_df.reset_index()