from transformers import TrainerCallback
import torch
from torch.utils.data import Dataset
import random

class AlpacaDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        if 'input' in item and item['input']:
            text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
        else:
            text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': encoding['input_ids'].copy()
        }

def generate_response(model, tokenizer, prompt, device, max_gen_len=500, stream_output=False):
    """Shared generation function used by both callback and evaluation
    
    Args:
        stream_output: If True, prints tokens as they're generated (for evaluation)
                      If False, returns complete response silently (for callback)
    """
    # Tokenize with attention mask
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    # attention_mask = inputs.attention_mask.to(device)
    
    past_key_values = None
    
    # Get initial output
    outputs = model(
        input_ids=input_ids, 
        # attention_mask=attention_mask,
        past_key_values=past_key_values, 
        mode="eval"
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    
    # Generate remaining tokens
    for _ in range(max_gen_len - 1):
        # new_attention_mask = torch.ones(1, 1, device=device)
        
        outputs = model(
            input_ids=pred_token_idx, 
            # attention_mask=new_attention_mask,
            past_key_values=past_key_values, 
            mode="eval"
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        
        # Streaming output for evaluation
        if stream_output:
            # Decode incrementally for streaming
            generated_text = (
                tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=False,
                )
                .strip()
                .split(" ")
            )
            
            now = len(generated_text) - 1
            if now > pos:
                print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                pos = now
        
        if pred_token_idx.item() == tokenizer.eos_token_id:
            break
    
    # Final decoding
    if stream_output:
        # Print remaining tokens
        final_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False,
        ).strip().split(" ")
        if pos < len(final_text):
            print(" ".join(final_text[pos:]), flush=True)
    
    # Return full response
    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
        spaces_between_special_tokens=False,
    )
    
    return response, generated_ids

class GenerationMonitorCallback(TrainerCallback):
    """Callback to monitor generation quality during training - silent generation"""
    
    def __init__(self, tokenizer, raw_dataset, device, log_steps=50, num_samples=3, max_gen_len=500):
        self.tokenizer = tokenizer
        self.device = device
        self.log_steps = log_steps
        self.max_gen_len = max_gen_len
        self.monitor_examples = []
        for i in range(min(num_samples, len(raw_dataset))):
            self.monitor_examples.append(raw_dataset[i])
        
        print(f"Initialized callback with {len(self.monitor_examples)} monitoring examples")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.log_steps == 0 and state.global_step > 0:
            self.generate_samples(state.global_step, kwargs.get('model'))
    
    def generate_samples(self, step, model):
        if model is None:
            return
            
        model.eval()
        print(f"\n{'='*80}")
        print(f"Step {step} - Generated Samples (silent generation)")
        print('='*80)
        
        with torch.no_grad():
            for i, example in enumerate(self.monitor_examples):
                # Create prompt
                instruction = example['instruction']
                input_text = example['input']
                
                if input_text and len(input_text.strip()) > 0:
                    prompt = f"{instruction}\n\n{input_text}"
                else:
                    prompt = instruction
                
                formatted_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"
                
                # Generate silently (stream_output=False)
                response, _ = generate_response(
                    model, self.tokenizer, formatted_prompt, 
                    self.device, self.max_gen_len, 
                    stream_output=False  # SILENT - no streaming
                )
                
                # Print only the final result
                print(f"\nSample {i+1}:")
                print(f"Instruction: {instruction[:100]}..." if len(instruction) > 100 else f"Instruction: {instruction}")
                if input_text and len(input_text.strip()) > 0:
                    print(f"Input: {input_text[:100]}..." if len(input_text) > 100 else f"Input: {input_text}")
                print(f"Generated: {response}")
                print(f"Expected: {example['output'][:200]}..." if len(example['output']) > 200 else f"Expected: {example['output']}")
                print("-" * 60)
        
        print('='*80)
        model.train()