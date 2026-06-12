
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import ast
import json

torch.random.manual_seed(0) 
model_id = "microsoft/Phi-3-mini-4k-instruct"
# model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# model_id = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
model = AutoModelForCausalLM.from_pretrained( 
    model_id,  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=False,  
) 

tokenizer = AutoTokenizer.from_pretrained(model_id) 
#offload to cpu

background_prompt = """
<|instruction|>
# TASK:
Using the provided caption give the best possible background
# ANSWER FORMAT:
- best possible background for the even that happening on the caption
# TIPS:
- The background should be a single word or a short phrase.
- The background should be relevant to the caption.

<|end of instruction|>

## Example: 
Caption: "a blue bird and a brown bear."
Background: "a forest."
"""

position_prompt = """
<|instruction|>
# TASK:
position grid = [top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right]
Find the suitable position from the image position grid for the provided caption and list of objects.
# ANSWER FORMAT:
- only a dictionary with object name as key and position as value.
# TIPS:
- each object should have a position
- the position should be from the given position grid.
- output should only be a dictionary with object name as key and position as value.
- do not include any explanation or additional text.
<|end of instruction|>
## Example: 
Caption:"a blue bird right to a brown bear."
Objects:["bird", "bear"]
Position:{"bird": "top-right", "bear": "bottom-left"}
Caption:"Two red cars parked at the roadside."
Objects:["car", "road"]
Position:{"car": "left", "road": "bottom"}
"""

side_prompt = """
<|instruction|>
# TASK:
Using the provided caption and object list state the direction of obj1 with respect to obj2.
# ANSWER FORMAT:
- list of two words indicating the direction of obj1 with respect to obj2.
# TIPS:
- first word corresponds to horizontal alignment.
- first word should be from ["left", "right", "same"]
- second word corresponds to vertical alignment.
- second word should be from ["above", "below", "same"]
- The side should be relevant to the caption.
- if the relationship alignment type is not present in the caption then use "same".

<|end of instruction|>

## Example: 
Obj1: "bear"
Obj2: "bird"
Caption: "a blue bird left to a brown bear."
Relation: ["right", "same"]
"""

relation_prompt = """
<|instruction|>
# TASK:
Using the provided caption and object list state the relation between obj1 and obj2.
# ANSWER FORMAT:
- Dictionary stating the relation of obj1 to obj2 and the corresponding subject and the object of relation.
- "s" for subject, "r" for relation and "o" for object.
# TIPS:
- first word corresponds to horizontal alignment.
- first word should be from ["left", "right"]
- second word corresponds to vertical alignment.
- second word should be from ["above", "below"]
- The relation should be relevant to the caption.
- if there is no relation between obj1 and obj2 then then leave the "r" empty.

<|end of instruction|>

## Example: 
Obj1: "bear"
Obj2: "bird"
Caption: "a blue bird left to a brown bear."
Relations: {"s":"bird", "r":"left", "o":"bear"}
"""
# grid_prompt = """
# <|instruction|>
# # TASK:
# position grid = [top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right]
# Form a suitable cluster of positions from the image position grid around the given object position. Use the provided caption and list of object position.
# # ANSWER FORMAT:
# - best possible cluster of positions created around the given position for each object in the list.
# # TIPS:
# - the position should be from the given position grid.
# - the position should be a cluster of positions.
# - consider the size of the objects when forming the cluster.
# - clster should be dense and not too sparse.
# <|end of instruction|>

# ## Example: 
# Objects: ["bird", "bear"]
# Position: {"bird": "top-right", "bear": "bottom-left"}
# Position cluster: {"bird": ["top-right", "right"], "bear": ["bottom-left", "bottom", "left", "center"]}
# """

grid_prompt = """
<|instruction|>
# TASK:
position grid = 000\n000\n000
Form a suitable cluster of positions from the  position grid around the given object position. Use the provided caption and list of object position.
# ANSWER FORMAT:
- best possible cluster of positions created around the given position for each object in the list.
# TIPS:
- the position should be marked as 1 in the given position grid.
- the position should be a cluster of positions.
- consider the size of the objects when forming the cluster.
- cluster should be dense and not sparse.
- do not include any explanation or additional text.
<|end of instruction|>
## Example: 
Objects: ["bird", "bear"]
Position: {"bird": 001\n000\n000, "bear": 000\n000\n100}
Position cluster: {"bird": 011\n011\n011, "bear": 110\n110\n110}
"""
 
size_prompt = """
<|instruction|>
# TASK:
Sort list of objects based on the size provided in the caption.
# ANSWER FORMAT:
- best possible size order for each object in the list.
# TIPS:
- The size should be sorted based on the size provided in the caption.
- give priority to the size provided in the caption before using the common knowledge.
- The size should be sorted in ascending order.
- sorted list should contain all the objects in the Objects.
- do not include any explanation or additional text.
<|end of instruction|>
## Example: 
Caption: "a blue bird and a brown bear. The bird is bigger than the bear."
Objects: ["bird", "bear"]
Sort Objects: ["bear", "bird"]
Caption: "red car car parked at the roadside."
Objects: ["car", "car", "road"]
Sort Objects: ["car", "car", "road"]
"""

attribute_prompt = """
<|instruction|>
# TASK:
find the attribute type for the given object based on the caption.
# ANSWER FORMAT:
- one word to describe the attribute type.
# TIPS:
- output should be one word.
<|end of instruction|>
## Example: 
Caption: "a blue bird and a brown bear. The bird is bigger than the bear."
Object: "bird"
Attribute: "blue"
Type: "color"
Caption: "wooden chair in the living room."
Object: "chair"
Attribute: "wooden"
Type: "material"
"""

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 
positions_list = ['top-left', 'top', 'top-right', 'left', 'center', 'right', 'bottom-left', 'bottom', 'bottom-right']
def generate_phi_response(caption,objects=[], type=1):
    if type == 0:
        messages = [
        {"role": "system", "content": position_prompt},
        {"role": "user", "content": f"Now let's find the best background for following objects:\n Caption: {caption}\n Objects: {objects}"},
    ]
    if type == 1:
        messages = [
        {"role": "system", "content": position_prompt},
        {"role": "user", "content": f"Now let's find the best positions for following objects:\n Caption: {caption}\n Objects: {objects}"},
    ]
    if type == 2:
        messages = [
        {"role": "system", "content": grid_prompt},
        {"role": "user", "content": f"Now let's find position clusters for following objects and positions:\n Objects: {caption}\n positions: {objects}"},
    ]
    if type == 3:
        messages = [
        {"role": "system", "content": size_prompt},
        {"role": "user", "content": f"Now let's sort the objects based on the size:\n Caption: {caption}\n Objects: {objects}"},
    ]
    if type == 4:
        messages = [
        {"role": "system", "content": attribute_prompt},
        {"role": "user", "content": f"Now let's find the type of the attribute describing the given object:\n Caption: {caption}\n Object: {objects[0]}\n Attribute: {objects[1]}"},
    ]
    if type == 5:
        messages = [
        {"role": "system", "content": side_prompt},
        {"role": "user", "content": f"Now let's find the side of the object based on the caption:\n Caption: {caption}\n Obj1: {objects[0]}\n Obj2: {objects[1]}\n"},
    ]
    if type == 6:
        messages = [
        {"role": "system", "content": relation_prompt},
        {"role": "user", "content": f"Now let's find the relation between the objects based on the caption:\n Caption: {caption}\n Obj1: {objects[0]}\n Obj2: {objects[1]}"},
    ]
    
    generation_args = { 
        "max_new_tokens": 500, 
        "return_full_text": False, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    output = pipe(messages, **generation_args) 
    
    if type == 0:
        clean_out = output[0]['generated_text'].replace('Background: ','').replace('"','') 
    if type == 1:
        clean_out=output[0]['generated_text'].replace('Position:','').lstrip()
        try:
            clean_out = ast.literal_eval(clean_out)
        except:
            # import pdb; pdb.set_trace()
            clean_out = clean_out.replace('\\n-:','')
            clean_out=clean_out[clean_out.index('{'):clean_out.index('}')+1].lstrip()
            clean_out = ast.literal_eval(clean_out)
        for i in clean_out:
            if clean_out[i] not in positions_list:
                clean_out[i]=clean_out[i].split('-')[0]

                
    if type == 2:
        clean_out=output[0]['generated_text'].replace('Position cluster:','')
        clean_out=clean_out[clean_out.index('{'):clean_out.index('}')+1].lstrip()
        clean_out = ast.literal_eval(clean_out)
    if type == 3:
        clean_out=output[0]['generated_text'].replace('Sort Objects:','').lstrip()
        try:
            clean_out = ast.literal_eval(clean_out)
        except:
            try:
                clean_out=clean_out[clean_out.index('['):clean_out.index(']')+1].lstrip()
                clean_out = ast.literal_eval(clean_out)
            except:
                clean_out=clean_out[clean_out.index('['):].split(',')[:len(objects)+1]
    if type == 4:
        clean_out=output[0]['generated_text'].replace('Type:','').lstrip()
        clean_out=clean_out.replace('"','')
        # import pdb; pdb.set_trace()
        # clean_out = ast.literal_eval(clean_out)
    if type == 5:
        
        clean_out=output[0]['generated_text'].replace('Relation:','').lstrip()
        try:
            clean_out = ast.literal_eval(clean_out)
        except:
            clean_out=clean_out[clean_out.index('['):clean_out.index(']')+1].lstrip()
            clean_out = ast.literal_eval(clean_out)
    if type == 6:
        clean_out=output[0]['generated_text'].replace('Relations:','').lstrip()
        try:
            clean_out = ast.literal_eval(clean_out)
        except:
            clean_out=clean_out[clean_out.index('{'):clean_out.index('}')+1].lstrip()
            clean_out = ast.literal_eval(clean_out)
    del output
    return clean_out