#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-

task_template = """Given task: 
```
{task}
```
> Evaluation result: {result}"""

# Format list of few shots
def format_few_shot_examples(examples):
    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            task_template.format(
                task=eg.value["task"],
                result=eg.value["label"],
            )
        )
    return "\n\n------------\n\n".join(strs)



# Format list of few shots
def format_few_shot_examples_solutions(examples):
    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            task_template.format(
                task=eg.value["task"],
                result=eg.value["solution"],
            )
        )
    return "\n\n------------\n\n".join(strs)


