from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

new_person: Person = {'name': 'Adi', 'age' = 26}
print(new_person)