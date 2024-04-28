import numpy as np
import streamlit as st
from modules import plot_map, plot_deforestation
from utils import forest_over_time
from utils import total_deforestation
from glob import glob
import plotly.express as px
import json
import os


if not os.path.exists('promises.json'):
    promises = {}
    with open('promises.json', 'w') as f:
        json.dump(promises, f)
with open('promises.json') as f:
    promises = json.load(f)


files = glob("data/app/S2A*")
file_names = [file.split("/")[-1] for file in files]
items = {file: {"deforrestation": total_deforestation(file, promises.get(file, None))} 
         for file in file_names}

# st.write(items)

# Initialize session state variables
if 'selected_item' not in st.session_state:
    st.session_state.selected_item = None

def view_item(item_name):
    """ Set the selected item in the session state. """
    st.session_state.selected_item = item_name
    
def parse_name(item_name):
    parts = item_name.split("_")
    return parts[5], parts[7], parts[8]

# # Main page
if st.session_state.selected_item is None:
    st.title("Suppliers")
    
    # checkbox filter non-compliant items
    filter_non_compliant = st.checkbox("Show non-compliant items")
    items_to_show = items
    if filter_non_compliant:
        items_to_show = {item_name: items[item_name] for item_name in items if items[item_name]["deforrestation"] > 1}
    for item_name in items_to_show:
        stats = items[item_name]
        code, h, w = parse_name(item_name)
        # stat_info = ", ".join(f"{k}: {v}" for k, v in items[item_name].items())
        col1, col2 = st.columns([4, 1])
        with col1:
            col11, col12 = st.columns([1, 1])
            with col11:
                st.write(f"{code} - {h}/{w}")
                deforr = f'{stats["deforrestation"]:.0f}%'
                if stats["deforrestation"] < 1:
                    st.write(f':green[all good]')
                elif stats["deforrestation"] < 10:
                    st.write(f':orange[{deforr} deforrestation]')
                else:
                    st.write(f':red[{deforr} deforrestation. High deforrestation rate!]')
                    
                # <hr>
            with col12:
                pass
                # df = forest_over_time(item_name)
                # # plotly
                # fig = px.line(df, 
                #               x=df.index, 
                #               y="Forest Coverage")
                # # remove ticks and labels
                # fig.update_xaxes(showticklabels=False)
                # fig.update_yaxes(showticklabels=False)
                # # small height
                # fig.update_layout(height=200)
                
                # st.plotly_chart(fig, use_container_width=True)
        # col1.write(f"{item_name}")
        
        col2.button("View", key=item_name, on_click=view_item, args=(item_name,))
        # st.write("---")
else:
    # Subpage for an item
    item_details = items[st.session_state.selected_item]
    code, h, w = parse_name(st.session_state.selected_item)
    st.title(f"{code} - Sector X: {h} - Sector Y: {w}")
    # st.write(f"Stats for this item:")
    # for stat, value in item_details.items():
    #     st.write(f"{stat}: {value}")
        
    plot_map(st.session_state.selected_item)
    
    st.write("# Forest Coverage Over Time")
    # plot_deforestation(st.session_state.selected_item)
    df = forest_over_time(st.session_state.selected_item)
    promise = int(np.ceil(df["Forest Coverage"].max()))
    if st.session_state.selected_item in promises:
        promise = promises[st.session_state.selected_item]
    promise = st.number_input("Stated Forest Coverage", value=promise)
    promises[st.session_state.selected_item] = promise
    items[st.session_state.selected_item]["deforrestation"] = total_deforestation(st.session_state.selected_item, promise)
    with open('promises.json', 'w') as f:
        json.dump(promises, f)
    
    if item_details["deforrestation"] > 1 and promise > df["Forest Coverage"].min():
        st.error(f'Deforrestation of {item_details["deforrestation"]:.0f}% detected', icon="🚨")


    # st.line_chart(df, height=300, y="Forest Coverage")
    # plot "Forest Coverage" and a vertical line for the promise using plotly
    fig = px.line(df, x=df.index, y="Forest Coverage")
    fig.add_hline(y=promise, line_dash="dash", line_color="red", annotation_text="Today")
    # height
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # st.line_chart(df, height=300, y="Deforestation")
    fig = px.line(df, x=df.index, y="Deforestation")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


    
    if st.button("Back to list"):
        st.session_state.selected_item = None
