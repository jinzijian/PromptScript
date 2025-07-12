import json
import os
import subprocess
from llm_clients import *
from task_sampler import *

class LLMQueryHandler:
    def __init__(self, model_name, api_key, model_provider, save_dir, output_file_prefix='resps'):
        self.model_name = model_name
        self.api_key = api_key
        self.model_provider = model_provider
        self.output_file_prefix = output_file_prefix
        self.responses = {'results': []}
        self.save_dir = save_dir
        self.progress = {}  # To store progress separately

        # Initialize LLM model based on the provider with adjusted parameters
        if model_provider == 'openai':
            self.llm_model = OpenAIClient(api_key=self.api_key, model=self.model_name)
        elif model_provider == 'anthropic':
            self.llm_model = AnthropicClient(api_key=self.api_key, model=self.model_name)
        elif model_provider == 'ollama':
            self.llm_model = OllamaLLM(model=model_name)
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")

    def load_progress(self, num_types, reset=False):
        # Load progress from 'progress.json'
        fpath = os.path.join(self.save_dir, 'progress.json')
        if not reset and os.path.exists(fpath):
            with open(fpath, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {}

        # Load responses
        output_file = os.path.join(self.save_dir, f'{self.output_file_prefix}.json')
        if not reset and os.path.exists(output_file):
            with open(output_file, 'r') as f:
                self.responses = json.load(f)
        else:
            self.responses = {'results': []}

        self.output_file = output_file
        self.num_types = num_types

        # Initialize progress for current model and num_types if not present
        model_progress_key = f'{self.model_provider}_{self.model_name}_{num_types}'
        if model_progress_key not in self.progress:
            self.progress[model_progress_key] = {
                'last_task_id': None,
                'last_combination_index': -1
            }
        self.current_progress = self.progress[model_progress_key]

    def save_progress(self):
        # Save progress to 'progress.json'
        self.progress[f'{self.model_provider}_{self.model_name}_{self.num_types}'] = self.current_progress
        fpath = os.path.join(self.save_dir, 'progress.json')
        with open(fpath, 'w') as f:
            json.dump(self.progress, f, indent=4)

        # Save responses to the output file
        with open(self.output_file, 'w') as f:
            json.dump(self.responses, f, indent=4)

    def query_llm(self, prompt):
        try:
            system_prompt = "You are a helpful assistant."

            if self.model_provider == 'openai':
                # For OpenAI models, include the role in the messages parameter
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                response = self.llm_model(messages=messages)
                return response

            elif self.model_provider == 'anthropic':
                # For Anthropic models, include the system prompt in the overall prompt
                full_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
                response = self.llm_model(prompt=full_prompt)
                return response

            elif self.model_provider == 'ollama':
                # For Ollama, adjust accordingly
                full_prompt = f"{system_prompt}\n{prompt}"
                response = self.llm_model(full_prompt)
                return response

            else:
                raise ValueError(f"Unsupported model provider: {self.model_provider}")

        except Exception as e:
            print(f"Error querying LLM: {e}")
            return None

    def process_tasks(self, task_sampler, template, reset_progress=False, first_k=None):
        """
        Process tasks for the specified num_types and model.
        """
        self.load_progress(self.num_types, reset=reset_progress)
        tasks = task_sampler.tasks  # Access tasks
        task_ids = sorted(tasks.keys(), key=int)
        if first_k is not None:
            task_ids = task_ids[:first_k]
        for task_id in task_ids:
            if self.current_progress['last_task_id'] and int(task_id) < int(self.current_progress['last_task_id']):
                continue  # Skip already processed tasks

            task = tasks[task_id]
            task_description = task['task']
            time_constraint_list = task.get('time_constraints', [])
            time_constraint = time_constraint_list[0] if time_constraint_list else 'No time constraint'

            # Generate all constraint combinations for the current num_types
            combinations = task_sampler.sample_constraints(task_id, self.num_types)
            if not combinations:
                continue  # Skip if no combinations are available

            total_combinations = len(combinations)
            start_combination_index = self.current_progress['last_combination_index'] + 1 if (
                self.current_progress['last_task_id'] == task_id
            ) else 0

            for combination_index in range(start_combination_index, total_combinations):
                combination = combinations[combination_index]
                constraints_texts = combination['constraint_texts']
                constraint_types = combination['constraint_types']
                constraint_ids = combination['constraint_ids']

                # Generate the query prompt
                prompt = generate_query(task_description, constraints_texts, time_constraint, template)
                # print(prompt)
                print(f"Processing Task ID {task_id}, Model {self.model_provider}:{self.model_name}, Num constraints={self.num_types}, Combo {combination_index+1}/{total_combinations}")

                # Query the LLM
                response = self.query_llm(prompt)
                if response is None:
                    # Save progress and exit
                    self.current_progress['last_task_id'] = task_id
                    self.current_progress['last_combination_index'] = combination_index - 1
                    self.save_progress()
                    print("Error occurred, progress saved.")
                    return

                # Collect difficulties for selected constraints
                difficulties = {}
                for ctype, cid in zip(constraint_types, constraint_ids):
                    difficulty_key = ''
                    if ctype == 'skill_constraints':
                        difficulty_key = 's1' if cid == 0 else 's2'
                    elif ctype == 'item_constraints':
                        difficulty_key = 'i1' if cid == 0 else 'i2'
                    elif ctype == 'environment_constraints':
                        difficulty_key = 'e1' if cid == 0 else 'e2'
                    if difficulty_key in task:
                        difficulties[difficulty_key] = task[difficulty_key]

                # Save the response along with required details
                result = {
                    'prompt': prompt,
                    'task_id': task_id,
                    # 'constraint_ids': constraint_ids,
                    # 'constraint_types': constraint_types,
                    'constraints': difficulties,
                    'response': response
                }
                self.responses['results'].append(result)

                # Update progress
                self.current_progress['last_task_id'] = task_id
                self.current_progress['last_combination_index'] = combination_index
                self.save_progress()

            # # Reset combination index after each task
            # self.current_progress['last_combination_index'] = -1
            # self.save_progress()