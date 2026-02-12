from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM, get_linear_schedule_with_warmup
import transformers
from torch.optim import AdamW
import torch
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
    
    model = AutoModelForMaskedLM.from_pretrained('kuleshov-group/mdlm-owt', trust_remote_code = True)
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2', trust_remote_code = True)
    optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=500, 
        num_training_steps=config['max_steps']
    )


    wandb.init(
        project="llada-finetuning",
        config={
            "learning_rate": config['learning_rate'],
            "temperature": config['temperature'],
            "alpha": config['alpha'],
            "dataset": config['dataset'],
            "remask_ratio": config['remask_ratio'],
            "mask_ratio": config['mask_ratio']
        }
    )

    dataset = load_dataset(config['dataset'], "en" ,streaming = True, split = "train")

    return model, tokenizer, optimizer, dataset, scheduler


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

def remask_tokens(logits, sampled_tokens, mask, output_starts, remask_ratio, mask_token_id, method = "confidence"):

    if method == "confidence":

        # get the top n% 
        probs = F.softmax(logits, dim = -1)
        sampled_probs = torch.gather(probs, dim = -1, index = sampled_tokens.unsqueeze(-1)).squeeze(-1)

        confidence = sampled_probs.clone()
        
        for b, output_start in enumerate(output_starts):
            # confidence[b, :output_start] = float('inf')
            confidence[b, mask[b] == False] = float('inf')

        masked_tokens = sampled_tokens.clone()

        if remask_ratio == "random":
            remask_ratio = torch.rand(1).item()

        for b, output_start in enumerate(output_starts):
            # num_to_mask = int(remask_ratio * (mask[b, output_start:].sum().item()))
            num_to_mask = int(remask_ratio * (mask[b, :].sum().item()))
            _, lowest_conf_idx = torch.topk(confidence[b], k=num_to_mask, largest=False)
            masked_tokens[b, lowest_conf_idx] = mask_token_id
        
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
def train_loop(model, tokenizer, optimizer, dataset, config, scheduler, device):
    MASK_TOKEN_ID = tokenizer.vocab_size

    print("Pad token:", tokenizer.pad_token_id)
    temperature = config['temperature']
    alpha = config['alpha']

    batch_size = config['batch_size']
    remask_ratio = config['remask_ratio']
    batch = []

    model.to(device)
    model.train()
    input_device = model.device 
    print(f"Model Input Device: {input_device}")

    print(f"Model requires_grad: {next(model.parameters()).requires_grad}")
    print(f"Model training mode: {model.training}")

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
            inst = ex['text']
            
            #tokenize
            inst_tokens = tokenizer(inst, return_tensors = "pt", truncation = True, max_length = 512).input_ids
            # output_tokens = tokenizer(output, return_tensors = "pt", truncation = True, max_length = 512).input_ids

            # full model input
            tokens = torch.cat([inst_tokens], dim = 1)
            all_tokens.append(tokens.squeeze(0))

            output_start = inst_tokens.shape[1]
            all_output_starts.append(output_start)
        

        tokens = pad_sequence(all_tokens, batch_first = True, padding_value = tokenizer.pad_token_id)
        tokens = tokens.to(input_device)
        batch_tokens = (tokens != tokenizer.pad_token_id).sum().item()
        total_tokens += batch_tokens
        
        # create mask
        mask = torch.zeros_like(tokens, dtype = torch.bool)

        # for each in batch, create a random mask starting from output_start
        for b, output_start in enumerate(all_output_starts):
            # mask rate
            if config["mask_ratio"] == "random":
               mask_rate = torch.rand(1).item()
            else:
                mask_rate = config["mask_ratio"]
            # mask[b, output_start:] = torch.rand(tokens[b, output_start:].shape) < mask_rate
            mask[b, :] = torch.rand(tokens[b, :].shape) < mask_rate

        masked_tokens = tokens.clone()
        # apply mask
        masked_tokens[mask] = MASK_TOKEN_ID
        

        # STEP 1: send through model
        # attention_mask = (tokens != tokenizer.pad_token_id).long()
        timesteps = torch.zeros(masked_tokens.shape[0], device=input_device)

        # print("Input", tokenizer.decode(masked_tokens[0]))
        # print("Input Len", len(tokenizer.decode(masked_tokens[0])))
        logits_1 = model(masked_tokens, timesteps=timesteps)
        output_device = logits_1.device
        target_tokens = tokens.to(output_device)
        mask_dev = mask.to(output_device)
        # basic model loss
        loss_1 = torch.nn.functional.cross_entropy(logits_1[mask_dev], target_tokens[mask_dev])
        # get entropy for unmasked and masked positions
        probs = F.softmax(logits_1, dim = -1)
        entropy_1 = -torch.sum(probs * torch.log(probs + 1e-10) , dim = -1) # (batch, seq_len)

        pad_mask = (tokens == tokenizer.pad_token_id)
        entropy_1 = entropy_1.masked_fill(pad_mask, 0.0)

        # get avg entropy
        masked_entropy_1 = entropy_1[mask].mean().item()

        output_mask = torch.zeros_like(tokens, dtype=torch.bool)
        for b, start in enumerate(all_output_starts):
            # output_mask[b, start:] = True
            output_mask[b, :] = True
        unmasked_output = ((output_mask & ~mask) & ~pad_mask)
        unmasked_entropy_1 = entropy_1[unmasked_output].mean().item() if unmasked_output.any() else 0.0 

        # Prepare for STEP 2
        # sample
        logits_with_noise = add_gumbel_noise(logits_1.detach(), temperature = temperature)
        sampled_tokens = torch.argmax(logits_with_noise, dim=-1)
        # reset prompt
        # for b, start in enumerate(all_output_starts):
        #     sampled_tokens[b, :start] = tokens[b, :start]

        # remask
        masked_tokens = remask_tokens(logits_1, sampled_tokens, mask, all_output_starts, remask_ratio, MASK_TOKEN_ID)

        # STEP 2: send through the model
        logits_2 = model(masked_tokens, timesteps=timesteps)
        # model loss with the original mask
        loss_2 = torch.nn.functional.cross_entropy(logits_2[mask.to(logits_2.device)], tokens.to(logits_2.device)[mask.to(logits_2.device)])

        # get entropy for unmasked and masked positions
        probs = F.softmax(logits_2, dim = -1)
        entropy_2 = -torch.sum(probs * torch.log(probs + 1e-10) , dim = -1) # (batch, seq_len)
        entropy_2 = entropy_2.masked_fill(pad_mask, 0.0)
        # get avg entropy masked
        masked_entropy_2 = entropy_2[mask].mean().item()
        # get avg entropy unmasked
        unmasked_entropy_2 = entropy_2[unmasked_output].mean().item() if unmasked_output.any() else 0.0 

        # total loss
        loss = calculate_loss(loss_1, loss_2, method = "alpha", a = alpha)
        # backprop
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        with torch.no_grad():
            pred_2 = torch.argmax(logits_2, dim=-1)
            unmasked = (masked_tokens != MASK_TOKEN_ID) & output_mask
            unmasked = unmasked & ~pad_mask
            changed_step_2 = ((masked_tokens != pred_2) & unmasked).float().sum() / unmasked.float().sum()
            changed_step_1 = ((((sampled_tokens != tokens) & ~mask) & ~pad_mask) & output_mask).float().sum() / ((~mask & ~pad_mask) & output_mask).float().sum()            


        wandb.log({
            "loss": loss.item(),
            "loss_1": loss_1.item(),
            "loss_2": loss_2.item(),
            "loss_gap": (loss_2 - loss_1).item(),
            "masked_entropy_1": masked_entropy_1,
            "unmasked_entropy_1": unmasked_entropy_1,
            "masked_entropy_2": masked_entropy_2,
            "unmasked_entropy_2": unmasked_entropy_2,
            "entropy_gap": (masked_entropy_2 - masked_entropy_1),
            "tokens_in_batch": batch_tokens,
            "total_tokens": total_tokens,
            "grad_norm": grad_norm,
            "unmasked_revision_rate_1": changed_step_1.item(),
            "unmasked_revision_rate_2": changed_step_2.item(),
            "step": i
        })

        batch = []  # reset batch
        if i >= config['max_steps']:
            break
    
    wandb.finish()
    torch.cuda.empty_cache()

def save_model(model, save_path):
    # save model
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaDA model")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the config file")

    # load args
    args = parser.parse_args()
    config = load_config(args.config_path)
    print("Config loaded from", args.config_path)
    model, tokenizer, optimizer, dataset, scheduler = initialize(config)
    tokenizer.pad_token = tokenizer.eos_token
    print("Initialization done.")
    # print("Model, optimizer, and dataset prepared.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Starting training loop...")
    # print(f"Tokenizer mask: {tokenizer.mask_token_id}")
    
    train_loop(model, tokenizer, optimizer, dataset, config, scheduler, device=device)

    print("Training completed. Saving model...")
    save_model(model, save_path = "./finetuned_llada")
    print("Model saved to ./finetuned_llada")

