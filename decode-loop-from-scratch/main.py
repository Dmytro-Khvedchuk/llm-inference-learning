from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import Tensor

# the model that we are using
model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
# model_name: str = "google/gemma-4-31B"

temperature_value: float = 0.8
top_k: int = 5
top_p: float = 0.7

# the model object initialization, it holds weights and other things.
model = AutoModelForCausalLM.from_pretrained(model_name)

# the tokenizer for our model, holds the vocabulary of each token. Basically vocab is {token: id}
tokenizer = AutoTokenizer.from_pretrained(model_name)

class MyFirstLLMModel:

    def __init__(self, input_prompt: str) -> None:
        self.input_prompt = input_prompt
        self.logits = None
        self.past_key_values = None
        self.output_sequence: list = []
        self.next_input = None

    def _top_k_modification(self, logits: Tensor) -> Tensor:
        top_k_values: Tensor = torch.topk(logits, top_k, dim=-1).values
        return torch.masked_fill(logits, logits <= top_k_values[:, -1], float("-inf"))
        
    def _top_p_modification(self, logits: Tensor) -> Tensor:

        
        # take the sorted values and indicies from the logits to get the most probable eight at the start.
        sorted_values, sorted_indicies = torch.sort(logits, descending=True)
        
        # normalize the probabilities by the softmax, so they will sum up to 1.
        probs: Tensor = torch.softmax(sorted_values, dim=-1)
        
        # calculate the CDF for the probs.
        cumulative_probs: Tensor = torch.cumsum(probs, dim=-1)
        
        # get the mask only of the probs, that are larger then the threshold
        sorted_mask: Tensor = cumulative_probs > top_p

        # not quite sure, reversing the second dimension?
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()

        # mapping the fields as False
        sorted_mask[..., 0] = False

        # creating a tensor of zeros with dimensions of the sorted mask
        original_mask = torch.zeros_like(sorted_mask, dtype=torch.bool)

        # i don't know what sctter does.
        original_mask.scatter_(-1, sorted_indicies, sorted_mask)

        # return the logits with -inf at 0s, so they will not affect the prediction.
        return logits.masked_fill(original_mask, float("-inf"))

    def _temperature_application(self, logits: Tensor) -> Tensor:
        
        # optional: top_k
        # logits: Tensor = self._top_k_modification(logits)

        # optional: top_p
        logits: Tensor = self._top_p_modification(logits)

        logits: Tensor = logits / temperature_value
        probs: Tensor = torch.softmax(logits, dim=-1)
        next_token: Tensor = torch.multinomial(probs, num_samples=1)
        return next_token


    # the prefill function is for a first prompt sequence generation
    # it basically takes all of the sequence of prompts and creates the k, v and q values
    # and generates the next token.
    def _prefill(self):
        # on this step we are feeding out input prompt to get a sequence of token_ids from
        # our vocabulary, we use the return_tensors="pt" so it will return the pytorch tensor.
        data = tokenizer(self.input_prompt, return_tensors="pt") # -> (46, 123, 5325)

        # this is the forward pass of our model for our input ids tensor
        model_out = model(data.input_ids) # -> k, q, v

        # here we are getting the logits
        self.logits = model_out.logits # logit -> 

        # and here we are getting the keys and values, this is our KV-cache
        # we store them so we will not need to recompute it for each new token.
        self.past_key_values = model_out.past_key_values # kv

        # here we are taking the logit of only the last token that was generated.
        # basically logits right now have a size of (1, 6, vocab_size)
        # we are taking the last value of the 2nd dimension since it is our generated token
        # and on the output logits is (1, vocab_size), it basically have the score for each token
        logits = self.logits[:, -1, :] # (Batch, Time, vocab_size)

        next_token = self._temperature_application(logits)
        # next_token = torch.argmax(logits, dim=-1)
        # token_id

        # here we just taking the id, map it and get the str value of the token based on the id
        output = tokenizer.decode(next_token)

        # Actually im not sure about how we store the value and need some deeper understanding here
        # We are saving the value of the last token id. 
        self.next_input = next_token

        # we are saving the last token into our response.
        self.output_sequence = output

    # so it is as i understand called decode and it is memory-bandwidth bound because
    # we need to load all of the keys and values, but the computation is easy
    # and for large models, we can loads thousands of tokens for a single token.
    # and compute here is cheap beacuse its one matmul

    # the prefill on the other hand is compute bound because we only doing that computing the logits
    # for each token in the prompt sequence
    def generate(self, tokens: int = 100) -> str:
        if self.logits is None:
            self._prefill()

        for _ in range(tokens):

            # here we are doing the same as in the prefill
            # only we are passing the embed of last token we produced and keys and values
            # so we will not recompute it once again (kv-cache)
            model_out = model(
                self.next_input, # token id
                past_key_values=self.past_key_values
            )

            # same as in prefill
            self.logits = model_out.logits
            self.past_key_values = model_out.past_key_values

            logits = self.logits[:, -1, :]

            next_token = self._temperature_application(logits)
            # next_token = torch.argmax(logits, dim=-1)

            output = tokenizer.decode(next_token)

            self.output_sequence += output

            self.next_input = next_token

            print("".join(self.output_sequence))

            if next_token == tokenizer.eos_token_id:
                break

        return self.output_sequence


input_prompt: str = "Is this how it works?"

class_model = MyFirstLLMModel(input_prompt)

print(class_model.generate(1000))