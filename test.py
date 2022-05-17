import json
def getSampleSanitizedDataset( ):

    sample = open("data\sample_for_dev.json", "r")
    list = json.load(sample)
    sample.close() 
    
    for i in range(len(list)):   
        task_id = list[i]['task_id']
        task_description = list[i]['task_description']
        prompt = list[i]['prompt']
        test_list = list[i]['test_list']
        print(prompt)
import time
b=2
a = {"a":4, "b": b}
for i in range(1,4):
    b = b+1
    print(a)