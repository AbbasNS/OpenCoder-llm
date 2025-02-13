import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor, as_completed

from benchmark.base import Benchmark
from sanitize import sanitize
from eval.execution import check_correctness
from utils import refine_text, stream_jsonl

class MBPPPlus(Benchmark):

    name: str = "MBPPPlus"
    path: str = os.path.abspath(os.path.join(ROOT, "../data/MBPPPlus.jsonl"))

    general_stop_words = [  "<|endoftext|>",
                            "<|endofmask|>",
                            "</s>",
                            "\nif __name__",
                            "\ndef main(",
                            "\nprint(",
                            '\n```\n']
    
    completion_stop_words = [   "\ndef ",
                                "\nclass ",
                                "\nimport ",
                                "\nfrom ",
                                "\nassert " ]

    imports = [ "import math",
                "import re",
                "import sys",
                "import copy",
                "import datetime",
                "import itertools",
                "import collections",
                "import heapq",
                "import functools",
                "import hashlib",
                "import numpy",
                "import numpy as np",
                "import string",
                "from typing import *",
                "from collections import *"]

    def __init__(self,
                 name: str = "MBPPPlus",
                 num_samples: int = 1,
                 num_workers: int = 16,
                 timeout: float = 3.0,
                 prompt_type: str = "Instruction"):
        
        super().__init__()
        self.name = name
        self.num_samples = num_samples
        self.num_workers = num_workers
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
            
            tasks[task_id] = task_data
        
        return tasks
    
    def format_prompt(self, 
                     promblem: str,
                     test: str,
                     ) -> str:
        promblem = f"You are an expert Python programmer, and here is your task:\n{promblem}\n"
        test = f"Write Python code that pass the following test:\n```python{test}```\n"
        prompt = promblem + test
        return prompt
    
    def get_prompt(self):
        """
        Builds the prompt for the LM to generate from.
        """

        assert self.prompt_type == "Instruction", "Prompt type must be Instruction for MBPP"

        prompts = []
        for task_id, task_data in self.tasks.items():
            prompts.append(
                dict(
                    task_id = task_id,
                    prompt = refine_text(self.format_prompt(task_data["text"], task_data["test_list"][0]))
                )
            )
        return prompts

    def postprocess_generation(self, generation):
        """
        Postprocess the generations.
        """
        try:
            solution = sanitize(generation['completion'])
        except Exception:
            solution = program_extract(generation['completion'], program="python", mode="all")
        
        return dict(
            task_id = generation['task_id'],
            completion_id = generation['completion_id'],
            solution = solution
        )
    
    
    def process_results(self, solution):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """

        task_data = self.tasks[solution['task_id']]

        if self.name == "MBPPPlus":
            test_code = "\n".join(task_data['test_imports']) + "\n\n" + task_data['test']
        elif self.name == "MBPPBase":
            test_code = "\n".join(task_data['test_imports']) + "\n\n" + "\n".join(task_data['test_list'])

        code =  (
            "\n".join(self.imports) + "\n"
            + solution['solution'] + "\n"
            + test_code
        )

        result = check_correctness(solution['task_id'],
                                   solution['completion_id'],
                                   code,
                                   self.timeout)
        
        return result