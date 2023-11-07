import streamlit as st
import pandas as pd
import numpy as np
import datetime
import importlib
import plotly.express as px
from sklearn.neighbors import LocalOutlierFactor 


def anomaly(dataframe):
    model_LOF = LocalOutlierFactor(n_neighbors=6)
    LOF_predictions = model_LOF.fit_predict(dataframe[['Price']])
    model_LOF_scores = model_LOF.negative_outlier_factor_

    dataframe['LOF_anomaly_scores'] = model_LOF_scores
    dataframe['LOF_anomaly'] = LOF_predictions
    return dataframe



st. set_page_config(layout="wide") 




items_name = {'Beef':['kg',1], 
            'Pork':['kg',1], 
            'Lamb':['kg',1], 
            'Whole chicken':['kg',1],
            'Bacon':['gr', 250],      
            'Frankfurter':['gr', 250],   


            'Shrimps':['gr', 250],  
            'Canned tuna in water or oil':['gr', 250],  
            'Eggs':['gr', 250],  
            'Butter':['gr', 250],  
            'Fresh milk':['gr', 250],  
            'Cheese':['gr', 250],  
            'Cooking oil':['gr', 250],  
            'Pre-packed sliced cooked HAM':['gr', 250],  
            'Japanese canned sardines':['gr', 250],  
            'Yoghurt, natural or plain':['gr', 250],  
            'Ice cream':['gr', 250],  
            'Olive oil, extra virgin':['gr', 250],  
            'Kamaboko':['gr', 250],  
            'Whipping Cream':['gr', 250]



            }



st.title('ICS Visualization')
excel = st.sidebar.file_uploader("Upload excel file")

def prepare_dataframe(file, items_name):
    df = pd.read_excel(file, 'JICS', skiprows=[0,1,2]).iloc[:, :15]
    df.columns = ['Index', 'Item class', 'Currency', 'Price 1', 'Qty 1', 'unit 1', 'Tax 1', 
                                                    'Price 2', 'Qty 2', 'unit 2', 'Tax 2', 
                                                    'Price 3', 'Qty 3', 'unit 3', 'Tax 3']

    df = df[0:36]  
    brand_df = df[df['Index'].isna()] 
    brand_dict = { 'Brand 1':brand_df['Price 1'].tolist(),
                   'Brand 2':brand_df['Price 2'].tolist(),
                   'Brand 3':brand_df['Price 3'].tolist()}


    df_final = df[df['Index'].notna()].reset_index(drop=True)
    for i in range(len(df_final)):
        for j,k in enumerate(items_name):
            if k in df_final['Item class'][i]: df_final['Item class'][i] = k

    df_final["Brand 1"] = brand_dict['Brand 1']
    df_final["Brand 2"] = brand_dict['Brand 2']
    df_final["Brand 3"] = brand_dict['Brand 3']


    return df_final

df = prepare_dataframe(excel, items_name)

#st.dataframe(df)

selected_item = st.sidebar.selectbox('Check item', df['Item class'].unique())
st.sidebar.write('Standard quantity = '+str(items_name.get(selected_item)[1]))
st.sidebar.write('Standard unit = '+items_name.get(selected_item)[0])
recalc = st.sidebar.button('Recalculate and plot')


df_selected = df.loc[df['Item class'] == selected_item].reset_index()

#st.dataframe(df_selected)
col1, col2, col3 = st.columns(3)





def recalculation(df):
    price = data['Price'][0]
    qty = data['Qty'][0]
    unit = data['unit'][0]
    tax = data['Tax'][0]

    if items_name.get(selected_item)[0] == 'kg':
        if unit == 'kg':
            price_new = price/qty
        if unit == 'gr':
            price_new = price/qty * 1000

    if items_name.get(selected_item)[0] == 'gr':
        if unit == 'gr':
            price_new = price/qty * items_name.get(selected_item)[1]
        if unit == 'kg':
            price_new = price*qty / 1000





    new_record = pd.DataFrame([{'Price':price_new, 'Qty':items_name.get(selected_item)[1], 
                                'unit':items_name.get(selected_item)[0], 'Tax':tax}], index=['Recalc']).round(3)


    return pd.concat([df, new_record])




k = []
outlet = []
brand = []
count = 0 
for i in range(1, 4):
    for j in range(len(df_selected)):
        price_survey = df_selected['Price ' + str(i)][j]
        qty_survey = df_selected['Qty ' + str(i)][j]
        unit_survey = df_selected['unit ' + str(i)][j]
        tax_survey = df_selected['Tax ' + str(i)][j]
        brand_survey = df_selected['Brand ' + str(i)][j]
        
        data = pd.DataFrame([[price_survey,qty_survey,unit_survey,tax_survey]], columns=['Price', 'Qty', 'unit', 'Tax'], index=['Survey'])

        ### Standaryzacja (przeliczenie)
        if recalc:
            data = recalculation(data)
            k.append( float( data.loc[data.index == 'Recalc']['Price'].values) )
            outlet.append(i)
            brand.append(brand_survey)

        ### Display
        if i is 1:
            with col1:
                st.write(f"""**{str(brand_survey).strip()}**""")
                st.data_editor(data, key=count)

        if i is 2:
            with col2:
                st.write(f"""**{str(brand_survey).strip()}**""")
                st.data_editor(data, key=count)

        if i is 3:
            with col3:
                st.write(f"""**{str(brand_survey).strip()}**""")
                st.data_editor(data, key=count)

        count += 1



plot_df = pd.DataFrame({'Price':k, 'Outlet':outlet, 'Brand':brand }).dropna()
plot_df["LOF_anomaly"] = 123

if not plot_df.empty:
    plot_df = anomaly(plot_df)
    


average = np.average(plot_df['Price'])
std = np.std(plot_df['Price'])
mediana = np.median(plot_df['Price'])


result = all(map(lambda x: x == plot_df['LOF_anomaly'][0], plot_df['LOF_anomaly']))

if result is True:
    color_scale = 'greens'
else:
    color_scale = [[0, 'red'], [1, 'green']]

## PLOTOWANIE
fig = px.scatter(
    plot_df,
    x="Outlet",
    y="Price",
    #size="Price",
    color="LOF_anomaly",
    color_continuous_scale=color_scale,
    hover_name="Brand",
    size_max=10,
    opacity=1,
    )

fig. update(layout_coloraxis_showscale=False)
fig.update_traces(mode='markers', marker_line_width=2, marker_size=10, )
fig.add_hline(y=average, line_dash="dash", line_color="green")
fig.add_hline(y=average + 1*std, line_dash="dash", line_color="purple")
fig.add_hline(y=average - 1*std, line_dash="dash", line_color="purple")



if recalc:    
    st.plotly_chart(fig, theme="streamlit", use_container_width=False)



    st.sidebar.write( 'Average price (SP) = ' + str(average.round(2)) + "  / " + str(items_name.get(selected_item)[1]) + str(items_name.get(selected_item)[0]) )
    st.sidebar.write( 'Median price (SP) = ' + str(mediana.round(2)) + "  / " + str(items_name.get(selected_item)[1]) + str(items_name.get(selected_item)[0]) )