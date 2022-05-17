from datetime import datetime
from datetime import date
# import datetime


current_time = today = str(date.today()) + " " +(datetime.now()).strftime("%H:%M:%S")
print( current_time )