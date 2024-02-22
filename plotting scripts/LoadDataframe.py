from tbparse import SummaryReader
import pandas as pd

def load_dataframe(log_dir):

    reader = SummaryReader(log_dir)

    df = reader.tensors

    # Rename
    df = df.rename(columns={'step': 'Epoch'})

    df = df.set_index(['Epoch'])

    # For each tag - there must be a column
    tags = df.loc[:, "tag"].unique()

    data = {}

    for tag in tags:
        mask = df["tag"] == tag
        
        df_tmp = df.loc[mask]
        
        new_tag = tag.replace("_", " ")

        data[new_tag] = df_tmp.value 

    df = pd.DataFrame(data)

    image_dict_arr = df['generated imgs'].dropna().apply(SummaryReader.tensor_to_image)
 
    num_epochs = df.to_numpy().shape[0]
   
    log_interval = int(num_epochs / len(image_dict_arr)) + 1


    for i, idx in enumerate(range(0, num_epochs, log_interval)):
     
        df.loc[idx, 'generated imgs'] = image_dict_arr.iloc[i]['image']

    return df