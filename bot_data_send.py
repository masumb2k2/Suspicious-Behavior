import requests
import telepot
from datetime import datetime
import pytz
import geocoder

#get current time and date
IST = pytz.timezone('Asia/dhaka')
raw_TS = datetime.now(IST)
curr_date=raw_TS.strftime("%d-%m-%y")
curr_time=raw_TS.strftime("%H:%M:%S")
#Bot Connectivity
bot_token='7798758169:AAHlEV_yJKnsd9YTPN-sdR05vBjYPE06f3E'
chat_id='-1002448159857'
bot = telepot.Bot(bot_token)
# Get the current location using geocoder's IP method
g = geocoder.ip('me')
latitude, longitude = g.latlng
google_maps_url = f"https://www.google.com/maps?q={latitude},{longitude}"



bot.sendMessage(chat_id, f"VIOLENCE ALERT!! \nLOCATION: {google_maps_url} \nDate: {curr_date} \nTIME: {curr_time}")
bot.sendPhoto(chat_id, photo=open('image.png', 'rb'))