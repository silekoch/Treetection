import streamlit as st
from modules import plot_map, plot_deforestation
from utils import forest_over_time
from utils import total_deforestation
from glob import glob
import plotly.express as px


files = glob("data/app/S2A*")
file_names = [file.split("/")[-1] for file in files]
items = {file: {"deforrestation": total_deforestation(file)} 
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
    st.title("Item List")
    for item_name in items:
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
    st.write(f"Stats for this item:")
    for stat, value in item_details.items():
        st.write(f"{stat}: {value}")
        
    plot_map(st.session_state.selected_item)
    plot_deforestation(st.session_state.selected_item)
    
    if st.button("Back to list"):
        st.session_state.selected_item = None
