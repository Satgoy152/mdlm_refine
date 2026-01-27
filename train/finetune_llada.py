from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM
from torch.optim import AdamW
import torch
from accelerate import Accelerator
import wandb
import yaml
from datasets import load_dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import argparse
from tqdm import tqdm

# load from config file

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def initialize(config):
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map = 'auto')
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code = True)
    # model = AutoModelForMaskedLM.from_pretrained('kuleshov-group/mdlm-owt', trust_remote_code = True)
    # tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2', trust_remote_code = True)
    optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']))


    wandb.init(
        project="llada-finetuning",
        config={
            "learning_rate": config['learning_rate'],
            "temperature": config['temperature'],
            "alpha": config['alpha'],
            "dataset": config['dataset']
        }
    )

    dataset = load_dataset(config['dataset'], streaming = True, split = "train")

    return model, tokenizer, optimizer, dataset, accelerator


# accelerator
def prepare_model_optimizer_dataset(model, optimizer, dataset, accelerator):
    
    model, optimizer, dataset = accelerator.prepare(model, optimizer, dataset)# move to device

    return model, optimizer, dataset, accelerator


# helpers
def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def remask_tokens(logits, sampled_tokens, output_start, output_starts, method = "confidence"):

    if method == "confidence":

        # get the top n% 
        probs = F.softmax(logits, dim = -1)
        sampled_probs = torch.gather(probs, dim = -1, index = sampled_tokens.unsqueeze(-1)).squeeze(-1)

        confidence = sampled_probs.clone()
        
        for b, output_start in enumerate(output_starts):
            confidence[b, :output_start] = float('inf')

        masked_tokens = sampled_tokens.clone()

        for b, output_start in enumerate(output_starts):
            num_to_mask = int(0.5 * (sampled_tokens.shape[1] - output_start))
            _, lowest_conf_idx = torch.topk(confidence[b], k=num_to_mask, largest=False)
            masked_tokens[b, lowest_conf_idx] = 126336
        
        # num of tokens to remask
        # num_to_mask = int(0.5 * (sampled_tokens.shape[1] - output_start))
        # lowest confident tokens are remasked so find index of them
        # _, lowest_confidence = torch.topk(confidence, k = num_to_mask, largest = False)

        #create a new mask
        # mask = torch.zeros_like(sampled_tokens, dtype=torch.bool)
        # mask.scatter_(dim = 1, index = lowest_confidence, value = True)

    
    # apply mask
    # masked_tokens = sampled_tokens.clone()
    # masked_tokens[mask] = 126336
    return masked_tokens

def calculate_loss(loss_1, loss_2, method = "alpha", a = 0.5):
    if method == "alpha":
        return (a * loss_1) + ((1 - a) * loss_2)

# training loop
def train_loop(model, tokenizer, optimizer, dataset, config, accelerator):
    temperature = config['temperature']
    alpha = config['alpha']

    batch_size = config['batch_size']
    batch = []

    # model.to(device)
    model.train()

    total_tokens = 0

    for i, example in enumerate(tqdm(dataset)):
        # create batch
        batch.append(example)
        if len(batch) < batch_size:
            continue
        
        # for each example in batch, tokenize instruction and output
        all_tokens = []
        all_output_starts = []
        for ex in batch:
            inst = ex['instruction']
            output =  ex["output"]
            
            #tokenize
            inst_tokens = tokenizer(inst, return_tensors = "pt", truncation = True, max_length = 512).input_ids
            output_tokens = tokenizer(output, return_tensors = "pt", truncation = True, max_length = 512).input_ids

            # full model input
            tokens = torch.cat([inst_tokens, output_tokens], dim = 1)
            all_tokens.append(tokens.squeeze(0))

            output_start = inst_tokens.shape[1]
            all_output_starts.append(output_start)
        

        tokens = pad_sequence(all_tokens, batch_first = True, padding_value = tokenizer.pad_token_id)
        tokens = tokens.to(accelerator.device)
        batch_tokens = (tokens != tokenizer.pad_token_id).sum().item()
        total_tokens += batch_tokens
        
        # create mask
        mask = torch.zeros_like(tokens, dtype = torch.bool)

        # for each in batch, create a random mask starting from output_start
        for b, output_start in enumerate(all_output_starts):
            # mask rate
            mask_rate = torch.rand(1).item()
            mask[b, output_start:] = torch.rand(tokens[b, output_start:].shape) < mask_rate

        masked_tokens = tokens.clone()
        # apply mask
        masked_tokens[mask] = 126336

        # STEP 1: send through model
        logits_1 = model(masked_tokens).logits
        # basic model loss
        loss_1 = torch.nn.functional.cross_entropy(logits_1[mask], tokens[mask])
        # get entropy for unmasked and masked positions
        probs = F.softmax(logits_1, dim = -1)
        entropy_1 = -torch.sum(probs * torch.log(probs + 1e-10) , dim = -1) # (batch, seq_len)

        # get avg entropy
        masked_entropy_1 = entropy_1[mask].mean().item()

        output_mask = torch.zeros_like(tokens, dtype=torch.bool)
        output_mask[:, output_start:] = True
        unmasked_output = output_mask & ~mask
        unmasked_entropy_1 = entropy_1[unmasked_output].mean().item() if unmasked_output.any() else 0.0 

        # Prepare for STEP 2
        # sample
        logits_with_noise = add_gumbel_noise(logits_1.detach(), temperature = temperature)
        sampled_tokens = torch.argmax(logits_with_noise, dim=-1)
        # reset prompt
        sampled_tokens[:, :output_start] = tokens[:, :output_start]
        # remask
        masked_tokens = remask_tokens(logits_1, sampled_tokens, output_start, all_output_starts)

        # STEP 2: send through the model
        logits_2 = model(masked_tokens).logits
        # model loss with the original mask
        loss_2 = torch.nn.functional.cross_entropy(logits_2[mask], tokens[mask])

        # get entropy for unmasked and masked positions
        probs = F.softmax(logits_2, dim = -1)
        entropy_2 = -torch.sum(probs * torch.log(probs + 1e-10) , dim = -1) # (batch, seq_len)
        # get avg entropy masked
        masked_entropy_2 = entropy_2[mask].mean().item()
        # get avg entropy unmasked
        unmasked_entropy_2 = entropy_2[unmasked_output].mean().item() if unmasked_output.any() else 0.0 

        # total loss
        loss = calculate_loss(loss_1, loss_2, method = "alpha", a = alpha)
        # backprop
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # print(masked_entropy_1, unmasked_entropy_1, loss_1.item())

        wandb.log({
            "loss": loss.item(),
            "loss_1": loss_1.item(),
            "loss_2": loss_2.item(),
            "masked_entropy_1": masked_entropy_1,
            "unmasked_entropy_1": unmasked_entropy_1,
            "masked_entropy_2": masked_entropy_2,
            "unmasked_entropy_2": unmasked_entropy_2,
            "tokens_in_batch": batch_tokens,
            "total_tokens": total_tokens,
            "step": i
        })
        
        batch = []  # reset batch
    
    wandb.finish()
    torch.cuda.empty_cache()

def save_model(model, accelerator, save_path):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaDA model")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the config file")

    # load args
    args = parser.parse_args()
    config = load_config(args.config_path)
    print("Config loaded from", args.config_path)
    model, tokenizer, optimizer, dataset, accelerator = initialize(config)
    print("Initialization done.")
    model, optimizer, dataset, accelerator = prepare_model_optimizer_dataset(model, optimizer, dataset, accelerator)
    print("Model, optimizer, and dataset prepared.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Starting training loop...")
    print(f"Tokenizer mask: {tokenizer.mask_token_id}")
    train_loop(model, tokenizer, optimizer, dataset, config, accelerator=accelerator)

    print("Training completed. Saving model...")
    save_model(model, accelerator, save_path = "./finetuned_llada")
    print("Model saved to ./finetuned_llada")

