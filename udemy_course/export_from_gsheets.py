import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('test1_secret.json', scope)
client = gspread.authorize(creds)

sheet = client.open('test2').sheet1

python_gsheets = sheet.get_all_records()

import pandas as pd
import numpy as np

df = pd.DataFrame(python_gsheets)
df_id = df['id_user']

df_id.to_csv(r'C:\Users\pawel.nowik\Desktop\Dane\export_python_test_3.csv',index = None, header=True)
