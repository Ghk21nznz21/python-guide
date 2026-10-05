''' Script Description  '''

from collections import Counter
import csv 

# Note: params are using Typehint notation 
def fastLoop(numbers: list[float,]):
    ''' Efficient way to loop:
    1 - Avoiding count dot notation ()
    2 - Use numpy (uses C language) which is faster
    3 - Use generators (memory efficient)
    
    4 - Use libraries like zip, enumerate, and itertools
    '''
    
    # 1 
    squared_numbers_lambda = list(map(lambda x: x ** 2, numbers))
    squared_numbers = [x ** 2 for x in numbers] # list comprehension
    
    # 3
    def generator_example(numbers):
        # Extra Note: this is a inner function 
        # Just to show how to use generators
        for number in numbers:
            yield number ** 2
    
    return squared_numbers

def counter(my_object: list | dict | tuple | str ):
    ''' Efficient way to count occurrences of items in a list'''
    return Counter(my_object)

def dicts(dict1: dict, dict2: dict):
    ''' Efficient way to:
        - avoid errors if key does not exist
        - merge dictionaries
    '''
    value = dict1.get('c')  # Avoids errors if dosent exist 
    return {**dict1, **dict2} # merging dicts 
    
def multipleIfs(conditions: list[bool | str, ]):
    if all(conditions):
        print("All conditions are true")
    if any(conditions):
        print("At least one condition is true")
    match conditions[0]:
        case 'true':
            print('Its true')
        case 'false':
            print('Its false')
        case _:
            print('Its others')
    return 

def reduceCode(x: int): 
    ''' Efficient way to reduce code: '''
    # A == 1 if x is not false, empty or None 
    a,b = 1, 2 if x  else 0, 0
    
    # List comprehension 
    myList = [v if v < 0 else 0 for v in [a,b]]
    
    # Lambda --> function comprehension        
    return lambda x, y: x*y if x!=0 else 0

def acessVarByName():
    ''' Allow to acess variable by name '''
    for i in range(3):
        # instead of [model0, ..., model2]
    	print(globals()[f'model{i}'])
     
     
def acessClassFuncByName(class_instance: object, name: str):
    ''' Allow to acess functions in a class name '''
    func = getattr(class_instance, name)
    # run with func()
    return func 
	
def csv_multiple_writes():
    with open('behance/users.csv', 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        
def delete_multiple_indexs(my_list: list, indices_to_delete: list[int, ]):
    ''' Delete Multiple indexs in a list without shifting the indexs '''
	# Sort the indices in reverse order to avoid index shifting 
    my_list.sort(reverse=True)
    for index in indices_to_delete:
        del my_list[index]
    return my_list
    	
     