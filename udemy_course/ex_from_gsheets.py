import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('ex_gsheets_creds.json', scope)
client = gspread.authorize(creds)

sheet = client.open('structure_of_sales_department')

python_gsheets = sheet.worksheet('str_sellers').get_all_records()

import pandas as pd

df = pd.DataFrame(python_gsheets)
df.to_csv(r'C:\Users\pawel.nowik\Desktop\Python GSheets\str_sellers.csv',index = None, header=True)
