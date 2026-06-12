import argparse
import os
import math
import torch
from test_pipeline import ASQLDiffusionPipeline
from tqdm import tqdm
import json
import csv
from torchvision.utils import make_grid,save_image
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import random

def main(
    prompts,
    sg,
    seeds,
    output_directory,
    model_path,
    step_sizes,
    attn_res,
    gpu,
    dist,
    lambda_ours,
):
    pipe = load_model(model_path, gpu)
    pipe.dist = dist
    pipe.lambda_ours = lambda_ours
  
    errors=[]
    count=0
   
    for prompt,graph in tqdm(zip(prompts, sg), total=len(prompts)):
       
        seeds_list = [int(seeds)]
        images = []
        for step_size in tqdm(step_sizes):
            for sdx, seed in enumerate(seeds_list):
                print(f'Running on: "{prompt}"', f"seed: {seed}")
                image,texts,masks = generate(pipe, prompt,graph, seed, step_size, attn_res)
                for i in range(len(image)):
                    save_hrs(image[i], count, prompt, output_directory)    
            count+=1
                
    for u in errors:
        print(f"Prompt: {u[0]}, Count: {u[1]}, Error: {u[2]}") 


def load_model(model_path, device=0):
    device = (
        torch.device(f"cuda:{device}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    pipe = ASQLDiffusionPipeline.from_pretrained(model_path).to(device)

    return pipe


def generate(pipe, prompt,graph, seed, step_size, attn_res):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    generator = torch.Generator(device.type).manual_seed(seed)
    result, texts, masks = pipe(
        prompt=[prompt,graph],
        generator=generator,
        attn_res=(int(math.sqrt(attn_res)), int(math.sqrt(attn_res))),
    )
    
    return result["images"],texts, masks


def save_hrs(image, i, prompt, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    img_name = str(i) + '_' + str(1) + '_' + prompt.replace(' ','_')+'.jpg'
    image.save(os.path.join(output_directory, img_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, default="A red cat above a blue car"
    )

    parser.add_argument(
        "--sg", type=str, default='{"entities": [{"head":"cat","quantity":"","id":0,"attributes":["red"]},{"head":"car","quantity":"","id":1,"attributes":["blue"]}], "relations": [{"subject": 0,"relation": "above","object": 1}]}'
    )

    parser.add_argument("--seed", type=str, default="42")

    parser.add_argument("--output_directory", type=str, default="./outputs")

    parser.add_argument(
        "--model_path",
        type=str,
        default='sd2-community/stable-diffusion-2-1',
        help="The path to the model (this will download the model if the path doesn't exist)",
    )

    parser.add_argument("--step_size", type=float, default=20.0, help="the step size")

    parser.add_argument(
        "--attn_res",
        type=int,
        default=576,
        help="The attention resolution (use 256 for SD 1.4, 576 for SD 2.1)",
    )

    parser.add_argument(
        "--gpu", type=int, default=0, help="The GPU to run the model on"
    )

    parser.add_argument(
        "--dist", type=str, default="cos", help="The distance loss"  # could be 'kl'
    )

    parser.add_argument(
        "--lambda_ours", type=float, default=0.5, help="The lambda for the ours loss"
    )
    parser.add_argument("--type", choices=['counting','spatial','color','size','texture','shape','position','single','mutiple'], default='counting')
    parser.add_argument("--file_save", type=str, default='OUTPUT')
    parser.add_argument("--data_type", choices=['HRS','Drawbench','T2I','GenEval','FID'], default='Drawbench')

    args = parser.parse_args()
         
    main(
        [args.prompt],
        [json.loads(args.sg)],
        args.seed,
        args.output_directory,
        args.model_path,
        [args.step_size],
        args.attn_res,
        args.gpu,
        args.dist,
        args.lambda_ours,
    )