from openai import OpenAI
import json
import os

# print(os.listdir())
path = os.path.dirname(os.path.realpath(__file__))

init_file = open(path + "/matrix_prompts/initial.txt", "r")
init_text = init_file.read()

create_file = open(path + "/matrix_prompts/create.txt", "r")
create_text = create_file.read()

crossover_file = open(path + "/matrix_prompts/crossover.txt", "r")
crossover_text = crossover_file.read()

mutation_file = open(path + "/matrix_prompts/mutation.txt", "r")
mutation_text = mutation_file.read()

reverse_file = open(path + "/matrix_prompts/reverse.txt", "r")
reverse_text = reverse_file.read()

def split_prompts(response_content):
    try:
        data = json.loads(response_content)
        strategies = data.get("strategy", [])
        if not isinstance(strategies, list):
            raise ValueError("Invalid format: 'strategy' should be a list.")
        return strategies
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []
    
def clean_code_output(response_content):
    cleaned = response_content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split('\n', 1)[-1] 
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit('\n', 1)[0] 
    return cleaned.strip()

class GPTModel():
    def __init__(self, api_key, model, temperature):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key)

    # @rate_limited(4)
    def initial_strategy(self):
        # print("Creating strategy")

        init_prompt = init_text
        # print(init_prompt)

        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "user", "content": init_prompt},
            ],
            max_tokens = 1024,
            temperature = self.temperature,
            stream = False
        )
        # print(response.choices[0].message.content)

        strategy = split_prompts(response.choices[0].message.content)
        return strategy
    
    # @rate_limited(4)
    def strategy_to_code(self, strategy):
        # print("Creating code...")

        strategy_text = "\n".join(strategy)
        create_prompt = create_text.format(strategy_text.strip())
        # print(create_prompt)
        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "user", "content": create_prompt},
            ],
            max_tokens = 1024,
            temperature = self.temperature,
            stream = False
        )

        # print(response.choices[0].message.content)
        code = clean_code_output(response.choices[0].message.content)
        return code

    # @rate_limited(4)
    def crossover(self, p1_strategy, p2_strategy):
        # print("Crossover...")
        p1_stra_text = "\n".join(p1_strategy)
        p2_stra_text = "\n".join(p2_strategy)
        crossover_prompt = crossover_text.format(p1_stra_text, p2_stra_text)
        # print(crossover_prompt)

        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "user", "content": crossover_prompt},
            ],
            max_tokens = 1024,
            temperature = self.temperature,
            stream = False
        )
        
        # print(response.choices[0].message.content)
        crossover_strategy = split_prompts(response.choices[0].message.content)
        return crossover_strategy
    
    # @rate_limited(4)
    def mutation(self, strategy):
        # print("Mutation...")

        stra_text = "\n".join(strategy)
        mutation_prompt = mutation_text.format(stra_text)
        # print(mutation_prompt)

        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "user", "content": mutation_prompt},
            ],
            max_tokens = 1024,
            temperature = self.temperature,
            stream = False
        )

        # print(response.choices[0].message.content)
        mutation_strategy = split_prompts(response.choices[0].message.content)
        return mutation_strategy
    
    # @rate_limited(4)
    def reverse(self, strategy):
        # print("Reverse...")

        stra_text = "\n".join(strategy)
        reverse_prompt = reverse_text.format(stra_text.strip())
        # print(reverse_prompt)

        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "user", "content": reverse_prompt},
            ],
            max_tokens = 1024,
            temperature = self.temperature,
            stream = False
        )

        # print(response.choices[0].message.content)
        reversed_strategy = split_prompts(response.choices[0].message.content)
        return reversed_strategy