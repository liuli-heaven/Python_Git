import datetime
import re

def extract_numbers(text):
    text_numbers = re.findall(r'\d+', text)
    if len(text_numbers) < 3 or len(text_numbers) > 6:
        print(f"date format error")
    return text_numbers

def numbers_to_ISO(numbers : list):
    """numbers is a list for string"""
    result = numbers[0]
    for i in range(2):
        result = result + '-' + numbers[i + 1].zfill(2)
    result = result + 'T'

    for str in numbers[3:]:
        result = result + str.zfill(2) + ':'
    result = result[:-1] + '+'
    result = result + "08:00"
    return result

def get_time(text):
    time_ = numbers_to_ISO(extract_numbers(text))
    result = datetime.datetime.fromisoformat(time_)
    return result

input_time = input("输入日期：\n")
result_time = get_time(input_time)
print(result_time.date())
print(result_time.strftime("%Y-%m-%d, %H:%M:%S"))
