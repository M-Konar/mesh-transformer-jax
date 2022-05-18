import json
def filter():
    stream = open("unfiltered_generation.json", "r")
    data = json.load(stream)
    stream.close()
    tasks = []
    c=0
    for task_sample in data['tasks'] :
        filtered = task_sample['completion']
        try:
            filtered = filtered[0:filtered.index("[END]")]
        except:
            c+=1
            continue

        tasks.append(
            {
                "task_id": task_sample["task_id"],
                "sample_id": task_sample["sample_id"],
                "task_description": task_sample["task_description"],
                "completion": filtered ,
                "test_list":task_sample["test_list"]
            }
            )

    data['tasks'] = tasks
    data['incomplete_generations'] = c
    
    stream = open("filtered.json", "w")
    stream.write(json.dumps(data))
    stream.close()

if __name__ == "__main__":
    filter()