from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import torch
from accelerate import Accelerator
import wandb
import yaml
from datasets import load_dataset
import torch.nn.functional as F
import argparse
import tqdm

# load from config file

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def initialize(config):

    model = AutoModelForCausalLM.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code = True)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])

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

    return model, tokenizer, optimizer, dataset


# accelerator
def prepare_model_optimizer_dataset(model, optimizer, dataset):
    accelerator = Accelerator()
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

def remask_tokens(logits, sampled_tokens, output_start, method = "confidence"):

    if method == "confidence":

        # get the top n% 
        probs = F.softmax(logits, dim = -1)
        sampled_probs = torch.gather(probs, dim = -1, index = sampled_tokens.unsqueeze(-1)).squeeze(-1)

        confidence = sampled_probs.clone()
        confidence[:, :output_start] = float('inf')

        # num of tokens to remask
        num_to_mask = int(0.5 * (sampled_tokens.shape[1] - output_start))
        # lowest confident tokens are remasked so find index of them
        _, lowest_confidence = torch.topk(confidence, k = num_to_mask, largest = False)

        #create a new mask
        mask = torch.zeros_like(sampled_tokens, dtype=torch.bool)
        mask.scatter_(dim = 1, index = lowest_confidence, value = True)

    
    # apply mask
    masked_tokens = sampled_tokens.clone()
    masked_tokens[mask] = 126336

def calculate_loss(loss_1, loss_2, method = "alpha", a = 0.5):
    if method == "alpha":
        return (a * loss_1) + ((1 - a) * loss_2)

# training loop
def train_loop(model, tokenizer, optimizer, dataset, config, device):
    temperature = config['temperature']
    alpha = config['alpha']

    model.to(device)
    model.train()

    for i, example in enumerate(tqdm(dataset)):
        inst = example['instruction']
        output =  example["output"]
        
        #tokenize
        inst_tokens = tokenizer(inst, return_tensors = "pt", truncation = True, max_length = 512).input_ids
        output_tokens = tokenizer(output, return_tensors = "pt", truncation = True, max_length = 512).input_ids

        # full model input
        tokens = torch.cat([inst_tokens, output_tokens], dim = 1).to(device)

        # create mask
        mask = torch.zeros_like(tokens, dtype = torch.bool)
        output_start = inst_tokens.shape[1]

        # mask rate
        mask_rate = torch.rand(1).item()
        mask[:, output_start:] = torch.rand(tokens[:, output_start:]) < mask_rate

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
        masked_tokens = remask_tokens(logits_1, sampled_tokens, output_start)

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
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({
            "loss": loss.item(),
            "loss_1": loss_1.item(),
            "loss_2": loss_2.item(),
            "masked_entropy_1": masked_entropy_1,
            "unmasked_entropy_1": unmasked_entropy_1,
            "masked_entropy_2": masked_entropy_2,
            "unmasked_entropy_2": unmasked_entropy_2,
            "step": i
        })

        if i >= 1: break

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
    model, tokenizer, optimizer, dataset = initialize(config)
    print("Initialization done.")
    model, optimizer, dataset, accelerator = prepare_model_optimizer_dataset(model, optimizer, dataset)
    print("Model, optimizer, and dataset prepared.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Starting training loop...")
    train_loop(model, tokenizer, optimizer, dataset, config, device)

    print("Training completed. Saving model...")
    save_model(model, accelerator, save_path = "./finetuned_llada")
    print("Model saved to ./finetuned_llada")

