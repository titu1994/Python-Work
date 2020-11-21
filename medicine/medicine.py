import numpy as np
import datetime

begin_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2021, 3, 1)
delta = end_date - begin_date
day_count = delta.days

print('Begin date : ', begin_date)
print('End Date : ', end_date)
print("Number of days : ", delta.days)
print()


def count_med_strips(name, per_day, per_strip, verbose=True):
    total_count = day_count * per_day
    number_of_strips = total_count // per_strip + 1

    if verbose:
        print('Medicine {:15s}: Count = {:7.2f} ({:0.2f} per day) | Number of strips = {:3.0f}'.format(
            name, total_count, per_day, number_of_strips
        ))

    return total_count, number_of_strips


def count_med_strips_with_available_count(name, per_day, per_strip, available_count, verbose=True):
    total_count, number_of_strips = count_med_strips(name, per_day, per_strip, verbose=False)
    total_count -= available_count
    number_of_strips = total_count // per_strip + 1

    currently_available_strips = available_count // per_strip

    if verbose:
        print('Medicine {:15s}: Count = {:7.2f} ({:0.2f} per day) | Currently Available Strips = {:4.2f} | '
              'Number of strips to buy = {:3.0f}'.format(
                name, total_count, per_day, currently_available_strips, number_of_strips
            ))

    return total_count, number_of_strips


""" Count basic medicines """

# count_med_strips(name='Levipil 500', per_day=2, per_strip=10)
# count_med_strips(name='Cardiace 5', per_day=3, per_strip=10)
# # count_med_strips(name='Azoran', per_day=3, per_strip=10)
# count_med_strips(name='HCQS 200', per_day=2, per_strip=15)
# count_med_strips(name='Omnacortil 2.5', per_day=3, per_strip=10)
# count_med_strips(name='Ecosporin 75', per_day=1, per_strip=10)
# count_med_strips(name='Shellcal HD', per_day=1, per_strip=15)
# count_med_strips(name='Osteofos 70', per_day=1. / 7, per_strip=4)

""" Count mediciation with available accounted for """
count_med_strips_with_available_count(name='Levipil 500', per_day=2, per_strip=10, available_count=0)
count_med_strips_with_available_count(name='Cardiace 5', per_day=3, per_strip=10, available_count=0)
count_med_strips_with_available_count(name='Azoran', per_day=3, per_strip=10, available_count=0)
count_med_strips_with_available_count(name='HCQS 200', per_day=2, per_strip=15, available_count=0)
count_med_strips_with_available_count(name='Omnacortil 2.5', per_day=2, per_strip=10, available_count=0)
count_med_strips_with_available_count(name='Ecosporin 75', per_day=1, per_strip=10, available_count=0)
count_med_strips_with_available_count(name='Shellcal HD', per_day=1, per_strip=15, available_count=0)
count_med_strips_with_available_count(name='Osteofos 70', per_day=1. / 7, per_strip=4, available_count=0)
