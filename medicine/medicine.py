import numpy as np
import datetime

begin_date = datetime.date(2018, 5, 1)
end_date = datetime.date(2019, 7, 1)
delta = end_date - begin_date
day_count = delta.days

print('Begin date : ', begin_date)
print('End Date : ', end_date)
print("Number of days : ", delta.days)
print()

def count_med_strips(name, per_day, per_strip):
    total_count = day_count * per_day
    number_of_strips = total_count // per_strip + 1
    print('Medicine %s : Count = %d (%0.2f per day) | Number of strips = %d' % (name, total_count, per_day, number_of_strips))

count_med_strips(name='Levipil 500', per_day=2, per_strip=10)
count_med_strips(name='Cardiace 5', per_day=3, per_strip=10)
count_med_strips(name='Azoran', per_day=3, per_strip=10)
count_med_strips(name='HCQS 200', per_day=1, per_strip=15)
count_med_strips(name='Omnacortil 2.5', per_day=1, per_strip=10)
count_med_strips(name='Ecosporin 75', per_day=1, per_strip=10)
count_med_strips(name='Shellcal HD', per_day=1, per_strip=15)
count_med_strips(name='Osteofos 70', per_day=1. / 7, per_strip=4)