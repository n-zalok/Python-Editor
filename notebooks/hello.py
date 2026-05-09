''' create a user '''

import datetime
import os

class User:
    def __init__(self, name):
        self.name = name

    def get_birthday(self, birthday):
        self.birthday = datetime.datetime.strptime(birthday, "%m/%d/%Y").date()

    def days_till_birthday(self):
        today = datetime.datetime.today()

        next_birthday = datetime.datetime(today.year, self.birthday.month, self.birthday.day)
        if next_birthday < today:
            next_birthday = datetime.datetime(today.year + 1, self.birthday.month, self.birthday.day)

        days_left = (next_birthday - today).days + 1
        return days_left

        
name = input("Enter your name: ")
BirthDay = input("Enter your birthday (mm/dd/yyyy): ")

user = User(name)
user.get_birthday(BirthDay)
days_left = user.days_till_birthday()

print(f"Hello, {user.name}! You were born on {user.birthday}. You have {days_left} days left until your birthday.")