def format_entity(entity):
    entity = entity.strip()
    if entity in ccode_dict.keys():
        entity = ccode_dict[entity]
    elif entity in inv_dict.keys():
        entity = inv_dict[entity]
    if "@" in entity:
        entity = entity.split("@")[0]
    if "_" in enitity:
        entity = " ".join(entity.split("_"))
    return entity


def extract_df(file_name):
    col_names = file_name.split("-")
    col_names_without_loc = [re.sub(r'\d+', '', s) for s in col_names]
    file_name = '{}/{}.csv'.format(triples_path, "-".join(col_names_without_loc))

    if not os.path.isfile(file_name):
        # in case filename is inverted, we read it and swap columns back
        file_name = '{}/{}-{}-{}.csv'.format(triples_path, col_names_without_loc[2], col_names_without_loc[1], col_names_without_loc[0])
        df = pd.read_csv(file_name)
        columns_titles = df.columns  # put column in the same order defined in file_name
        columns_titles = [columns_titles[2],columns_titles[1],columns_titles[0]]
        df=df.reindex(columns=columns_titles)
    else:
        df = pd.read_csv(file_name)

    col_names[1] = "relation"
    df.columns = col_names  # add location digits back into the file name

    return df
