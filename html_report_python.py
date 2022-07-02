import dominate
from dominate.tags import *
import pandas as pd

table_headers = ['index','value']

def create_page(df_stats, list_securities):
    doc = dominate.document(title = 'Report')

    #compute name to append at the name of the figure
    unique_name = ''
    for i in list_securities:
        unique_name += '_' + i
    
    #start to create the html file
    with doc.head:
        link(rel = 'stylesheet', href = 'style.css')
        style("""table, th, td {border:1px solid black;) """)
    with doc:
        with div(cls='container', style='display:inline-block;vertical-align:top;margin-top: 20px;'):
            with table(id = 'stats', cls = 'table of statistics', style="width:100%"):
                caption(h3('Stastic'))
                with thead():
                    with tr():
                        for table_head in table_headers:
                            th(table_head)
                with tbody():
                    for i in range(len(df_stats)):
                        with tr():
                            td(df_stats.iloc[i]['index'])
                            td(round(df_stats.iloc[i]['value'],4))

        with div(cls='container', style='display:inline-block;margin-top: 50px;'):
            img(src = "./img/temp/table_montly_returns" + unique_name + ".png", alt = 'Distribution plot' ,width="550", height="330")

        with div(cls='container', style='display:inline-block;margin-top: 50px;'):
            img(src = "./img/temp/distribution_returns" + unique_name + ".png", alt = 'Distribution plot' ,width="480", height="350")

        with div(cls='container', style='display:inline-block;vertical-align:top;margin-top: 50px;margin-left: 50px;'):
            img(src = "./img/temp/drawdowns_period" + unique_name + ".png", alt = 'Distribution plot', width="576", height="288")

        with div(cls='container', style='display:inline-block;margin-top: 50px;'):
            img(src = "./img/temp/under_water_plot" + unique_name + ".png", alt = 'Distribution plot' ,width="576", height="288")



    print(doc)


list_securities = ['SSO','UBT', 'UST', 'UGL', 'DIG']

unique_name = ''
for i in list_securities:
    unique_name += '_' + i
df = pd.read_csv('./data/temp/statistics_all_weather'+ unique_name +'.csv')

create_page(df, list_securities)


#To create the html file call from the shell (in the right directory) $pyhton3 html_with_python.py >> report.html  This command will create a report.html file in the same directory with the report of the strategy