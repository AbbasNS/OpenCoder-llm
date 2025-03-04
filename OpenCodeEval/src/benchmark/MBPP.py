import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

from typing import List

from benchmark.base import Benchmark
from sanitize import sanitize
from eval.execution import check_correctness
from utils import refine_text, stream_jsonl

class MBPP(Benchmark):

    name: str = "MBPP"
    path: str = os.path.abspath(os.path.join(ROOT, "../data/mbpp.jsonl"))

    few_shots_start = 1
    few_shots_end = 4

    test_start = 10
    test_end = 510

    def __init__(self,
                 name:str = "MBPP",
                 timeout:float = 9.0,
                 prompt_type:str = "Instruction"): 
        super().__init__()
        
        self.name = name
        self.timeout = timeout
        self.prompt_type = prompt_type

        self.tasks = self.get_task()
    
    def get_task(self):
        """
        Get the task data from the jsonl file into a dictionary.
        """

        tasks = {}
        
        for task_data in stream_jsonl(filename=self.path):
            task_id = int(task_data["task_id"])

            task_data['text'] = refine_text(task_data['text'])
            
            tasks[task_id] = task_data
        
        return tasks

    
    def fewshot_examples(self):

        few_shots = []

        for task_id, task_data in self.tasks.items():
            if task_id >= self.few_shots_start and task_id < self.few_shots_end:
                few_shots.append(task_data)
        
        return few_shots
    
    def format_prompt(self,
                      promblem: str,
                      tests: List[str],
                      code: str = None
                    ) -> str:
        promblem = f"You are an expert Python programmer, and here is your task:\n{promblem}"
        test = "\n".join(tests)
        test = f"Write Python code that pass the following test:\n```python{test}```\n"
        prompt = promblem + test
        if code:
            code = refine_text(code)
            code = f"\n```python\n{code}\n```\n"
            prompt = prompt + code
        return prompt
    
    def get_few_shots_prompts(self):
        
        few_shots_prompts = []
        for few_shot in self.fewshot_examples():
            few_shots_prompts.append(self.format_prompt(few_shot["text"], few_shot["test_list"], few_shot["code"]))

        return '\n'.join(few_shots_prompts)
    
    def get_prompt(self):
        """
        Builds the prompt for the LM to generate from.
        """

        assert self.prompt_type == "Instruction", "Prompt type must be Instruction for MBPP"

        #few_shots_prompts = self.get_few_shots_prompts()
        prompts = []

        for task_id, task_data in self.tasks.items():
            if task_id >= self.test_start and task_id < self.test_end:
                # temporary to avoid code extraction bug with many test cases
                # prompt = few_shots_prompts + '\n' + self.format_prompt(task_data["text"], task_data["test_list"])
                prompt = self.format_prompt(task_data["text"], [task_data["test_list"][0]])
                prompts.append({
                    'task_id': task_id,
                    'prompt': prompt
                })

        return prompts

    def postprocess_generation(self, generation):
        """
        Postprocess the generations.
        """

        return dict(
            task_id = generation['task_id'],
            completion_id = generation['completion_id'],
            solution = sanitize(generation['completion'])
        )
    
    def process_results(self, solution):
        """
        Takes the list of LM generations and evaluates them against the test cases
        """

        task_data = self.tasks[solution['task_id']]

        code =  (
                    "\n".join(self.imports) + "\n"
                    + task_data['test_setup_code'] + "\n"
                    + solution['solution'] + "\n"
                    + "\n".join(task_data['test_list'])
                )
        
        result = check_correctness(solution['task_id'],
                                   solution['completion_id'],
                                   code,
                                   self.timeout)
        
        return result
