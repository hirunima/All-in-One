import argparse
import os
import math
import torch
from test_pipeline import ASQLDiffusionPipeline
from tqdm import tqdm
import pandas as pd
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
       
        if args.data_type=='T2I':
            seeds_list = [int(seeds)+i for i in range(10)]
        elif args.data_type=='GenEval':
            seeds_list = [int(i) for i in seeds.split(",")]
        else:
            seeds_list = [int(seeds)]
        images = []
        for step_size in tqdm(step_sizes):
            for sdx, seed in enumerate(seeds_list):
                print(f'Running on: "{prompt}"', f"seed: {seed}")
                
                if args.data_type=='GenEval':
                    image,texts,masks = generate(pipe, prompt['prompt'],graph, seed, step_size, attn_res)
                else:
                    image,texts,masks = generate(pipe, prompt,graph, seed, step_size, attn_res)
                
                for i in range(len(image)):
                  
                    if args.data_type=='HRS':
                        save_hrs(image[i], count, prompt, output_directory)
                    elif args.data_type=='Drawbench':
                        save_image_pil(image[i], count, prompt, output_directory)
                    elif args.data_type=='T2I':
                        save_t2i(image[i], sdx, prompt, output_directory)
                    elif args.data_type=='GenEval':
                        save_geneval(image[i], count, prompt.to_dict(), output_directory)

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

def save_cluster_image(image, prompt, seed, output_directory):
     
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_name = f"{output_directory}/{prompt}.png"
    print(f"Saving image to {file_name}")
    image.save(file_name)

def save_image_pil(image, i, seed, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_name = f"{output_directory}/{i}.png"
    image.save(file_name)

def save_hrs(image, i, prompt, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    img_name = str(i) + '_' + str(1) + '_' + prompt.replace(' ','_')+'.jpg'
    image.save(os.path.join(output_directory, img_name))

def save_geneval(image, i, prompt, output_directory):

    outpath = os.path.join(output_directory, f"{i:05}")
    os.makedirs(outpath, exist_ok=True)

    with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
        json.dump(prompt, fp)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    list_files = os.listdir(sample_path)
    sample_count = len(list_files)

    img_name = os.path.join(sample_path, f"{sample_count:04}.png")
    print(f"Saving image to {img_name}")
    image.save(os.path.join(sample_path, img_name))

def save_t2i(image, i, prompt, output_directory):

    sample_path = os.path.join(output_directory, "examples/samples/")
    os.makedirs(sample_path, exist_ok=True)

    img_name = os.path.join(sample_path, f"{prompt}_{i:06}.png")
    image.save(img_name)

def load_geneval(json_pth,csv_pth):
    gt_data = pd.read_json(csv_pth, lines=True)
    meta = []
    syn_prompt = []
    sg = []
    with open(json_pth,'r') as f:
        reader = json.load(f)
        for r, row in enumerate(reader):
            get_prompt = gt_data.loc[r]
            meta.append(get_prompt)
            sg.append(row)
    return meta, sg

def load_t2i(json_pth,csv_pth):
    gt_data = pd.read_table(csv_pth, sep="\t", header=None, names=['meta_prompt'])
    meta = gt_data["meta_prompt"].tolist()
    sg = []
    with open(json_pth,'r') as f:
        reader = json.load(f)
        for r, row in enumerate(reader):
            sg.append(row)
    return meta, sg

def load_hrs(json_pth,csv_pth):
    gt_data = pd.read_csv(csv_pth).to_dict('records')
    meta = []
    syn_prompt = []
    sg = []
    with open(json_pth,'r') as f:
        reader = json.load(f)
        for r, row in enumerate(reader):
            sg.append(row)
    for sample in gt_data:
        meta.append(sample['meta_prompt'])
        syn_prompt.append(sample['synthetic_prompt'])
        
    return meta, syn_prompt, sg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, default="a checkered bowl on a red and blue table"
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


    if args.data_type=='HRS':
        if args.type=='counting':
            _,prompts,sg = load_hrs('./data_evaluate/HRS/counting_prompts_sg_relations.json','./data_evaluate/HRS/counting_prompts.csv')
        elif args.type=='spatial':
            prompts,_, sg = load_hrs('./data_evaluate/HRS/spatial_compositions_prompts_sg_relations.json','./data_evaluate/HRS/spatial_compositions_prompts.csv')
        elif args.type=='size':
            prompts,_, sg = load_hrs('./data_evaluate/HRS/size_compositions_prompts_sg_relations.json','./data_evaluate/HRS/size_compositions_prompts.csv')
        elif args.type=='color':
            prompts,_, sg = load_hrs('./data_evaluate/HRS/color_compositions_prompts_sg_relations.json','./data_evaluate/HRS/color_compositions_prompts.csv')

    elif args.data_type=='T2I':
        if args.type=='color':
            prompts,sg = load_t2i('./data_evaluate/t2i/color_val_sg_relations.json','./data_evaluate/t2i/color_val.txt')
        elif args.type=='texture':
            prompts,sg = load_t2i('./data_evaluate/t2i/texture_val_sg_relations.json','./data_evaluate/t2i/texture_val.txt')
        elif args.type=='shape':
            prompts,sg = load_t2i('./data_evaluate/t2i/shape_val_sg_relations.json','./data_evaluate/t2i/shape_val.txt')
        elif args.type=='spatial':
            prompts,sg = load_t2i('./data_evaluate/t2i/spatial_val_sg_relations.json','./data_evaluate/t2i/spatial_val.txt')

    elif args.data_type=='GenEval':
        prompts,sg = load_geneval('./data_evaluate/geneval/evaluation_metadata_sg_relations.json','./data_evaluate/geneval/evaluation_metadata.jsonl')
            

    main(
        prompts,
        sg,
        args.seed,
        args.output_directory,
        args.model_path,
        [args.step_size],
        args.attn_res,
        args.gpu,
        args.dist,
        args.lambda_ours,
    )