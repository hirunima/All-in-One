import itertools
from typing import Any, Callable, Dict, Optional, Union, List, Tuple
import ast
from copy import deepcopy
import torch
import cv2
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    EXAMPLE_DOC_STRING,
    rescale_noise_cfg,
)

from pipeline_stable_diffusion_attend_and_excite_self import (
    AttentionStore,
    AttendExciteAttnProcessor,
)
from selfstore import SelfAttentionStore

from torchvision.utils import save_image,make_grid
import numpy as np
import math
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from phi import generate_phi_response
from clustering_mash import create_grid, get_distance, grid_dual_points
from PIL import Image
import re
from scipy.ndimage import distance_transform_edt
import random
import difflib

logger = logging.get_logger(__name__)

class ASQLDiffusionPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: None,
        requires_safety_checker: bool = False
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker
        )

        self.subtrees_indices = None
        self.doc = None

    def _aggregate_and_get_attention_maps_per_token(self):
        attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"),
        )
        self_attention_maps = self.attention_store.aggregate_self_attention(
            from_where=("up", "down", "mid"),
        )
        
        attention_maps_list = _get_attention_maps_list(attention_maps=attention_maps)
        self_attention_maps_list = self_attention_maps.permute(2, 0, 1) * 100  # _get_attention_maps_list(attention_maps=self_attention_maps)
        return attention_maps_list, self_attention_maps_list

    @staticmethod
    def _update_latent(
        latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue
            
            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(
                attnstore=self.attention_store, 
                place_in_unet=place_in_unet
            )
         
        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    def parsered(self, graph, prompt, attn_res):
      
        num_obj=len(graph['entities'])
        graph_embeddings={
        }
        objects=[]
        objects_rel=[]
        object_attn={}
        attributes={}
        quantity_bool = False
        prompt_attr = prompt
        prompt_count = prompt
        graph_rel= deepcopy(graph)
        for i in graph['entities']:
                
            objects.append(i['head'])
            objects_rel.append([])
            if i['quantity'] != '':
                quantity_bool = True
                object_attn[i['head']]=[]

                objects_rel[-1].append(i['head'])
                object_attn[i['head']]=[i['head']]
                for a in i['attributes']:
                    object_attn[i['head']].append(a)
                for j in range(int(i['quantity'])-1):
                    objects_rel[-1].append(i['head'])
                   
            else:
                object_attn[i['head']]=[i['head']]
                objects_rel[-1].append(i['head'])
                # object_attn[i['head']].append(i['head'])
                for j in i['attributes']:
                    object_attn[i['head']].append(j)
         
        position_grid = {'top-left':(0,0), 'top':(1,0), 'top-right':(2,0), 'left':(0,1), 'center':(1,1), 'right':(2,1), 'bottom-left':(0,2), 'bottom':(1,2), 'bottom-right':(2,2)}
        
        try:
            position = generate_phi_response(prompt,objects,type=1)

        except Exception as e:
            print(f"Error generating position: {e}")
            position = {}

        # Ensure all values in 'position' are valid keys from position_grid
        for k, v in position.copy().items():
            if v not in position_grid.keys():
                position.pop(k, None)

        try:
            position_sorted = generate_phi_response(prompt,objects,type=3)
        except Exception as e:
            print(f"Error generating position_sorted: {e}")
            position_sorted = []

        # Combine object_attn keys, position keys, and position_sorted, removing duplicates
        combined_keys = list(dict.fromkeys(list(object_attn.keys()) + list(position.keys()) + list(position_sorted)))
        
        for i in combined_keys:
            if i not in position.keys():
                available_positions = [p for p in position_grid.keys() if p not in position.values()]
                position[i] = random.choice(available_positions) if available_positions else "center"
            if i not in position_sorted:
                position_sorted.append(i)
            if i not in object_attn.keys():
                object_attn[i] = [i]
        
        cluster_names = create_grid(position_sorted,attn_res)
        
        grid = get_distance(attn_res)

        
        reverse= {v: k for k, v in position_grid.items()}
        
        position_cluster = {k: np.ones((3, 3)) for k, v in cluster_names.items()}
        horizontal = ['right', 'left', 'same']
        vertical = ['above', 'below', 'same']
        
        for i in graph_rel['relations']:

            sub = i['subject']
            obj = i['object']
            relation = i['relation']
            if obj in position_cluster.keys() and sub in position_cluster.keys():
                cluster_fuzzy_obj = position_cluster[obj]
                cluster_fuzzy_sub = position_cluster[sub]
            else:
                continue
            sub_pos = position_grid[position[sub]]
            obj_pos = position_grid[position[obj]]
            
            #check for horizontal
            if relation in horizontal:
                if relation == 'right':
                    for p in range(3):
                        for q in range(0, obj_pos[0]+1):
                            cluster_fuzzy_sub[p][q] = 0
                        for q in range(sub_pos[0], 3):
                                cluster_fuzzy_obj[p][q] = 0
                elif relation == 'left':
                    for p in range(3):
                        for q in range(obj_pos[0], 3):
                            cluster_fuzzy_sub[p][q] = 0
                        for q in range(0, sub_pos[0]+1):
                            cluster_fuzzy_obj[p][q] = 0

            if relation in vertical:
                if relation == 'above':
                    for q in range(3):
                        for p in range(obj_pos[1], 3):
                            cluster_fuzzy_sub[p][q] = 0
                        for p in range(0, sub_pos[1]+1):
                            cluster_fuzzy_obj[p][q] = 0
                elif relation == 'below':
                    for q in range(3):
                        for p in range(0, obj_pos[1]+1):
                            cluster_fuzzy_sub[p][q] = 0
                        for p in range(sub_pos[1], 3):
                            cluster_fuzzy_obj[p][q] = 0

            cluster_fuzzy_sub[sub_pos[1]][sub_pos[0]] = 1
            cluster_fuzzy_obj[obj_pos[1]][obj_pos[0]] = 1
            position_cluster[obj] = cluster_fuzzy_obj
            position_cluster[sub] = cluster_fuzzy_sub

        print('objects:', position_cluster.keys())
        
        position_cluster,position,grid = grid_dual_points(cluster_names,position,objects_rel,attn_res,position_cluster)
        
        for key in position_cluster.keys():
            for q,quant in enumerate(position_cluster[key]):
                cluster_fuzzy = quant
                revert_fuzzy = []
                for m in range(3):
                    for n in range(3):
                        if cluster_fuzzy[m][n] == 1:
                            revert_fuzzy.append(reverse[(n,m)])

                position_cluster[key][q] = [revert_fuzzy]
                mask = np.zeros(attn_res)
                
                for p in position_cluster[key][q][0]:
                    coordinates = grid[p][:2]
                    
                    mask[coordinates[0][1]:coordinates[1][1],coordinates[0][0]:coordinates[1][0]] = 1
                position_cluster[key][q].append(position[key])
                position_cluster[key][q].append(mask)
            

        attributes['prompt'] = prompt
        attributes['prompt_attr'] = prompt
        graph_embeddings={'caption': prompt,'position_sorted': position_sorted, 'cluster_size': cluster_names, 'clusters':position_cluster, 'grid':grid, 'objects':object_attn,'quantity':quantity_bool, 'attributes':attributes}
        torch.cuda.empty_cache()
        return graph_embeddings
        
    # Based on StableDiffusionPipeline.__call__ . New code is annotated with NEW.
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        attn_res: Optional[Tuple[int]] = (16, 16),
        step_size: float = 20.0,
        parsed_prompt: str = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            attn_res (`tuple`, *optional*, default computed from width and height):
                The 2D resolution of the semantic attention map.
            step_size (`float`, *optional*, default to 20.0):
                Controls the step size of each Ebama update.
            parsed_prompt (`str`, *optional*, default to None):


        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )


        # 1. Check inputs. Raise error if not correct
        self.doc = self.parsered(prompt[1],prompt[0],attn_res)
        
        self.gaussiansmoothing = GaussianSmoothing()
        
        #single caption
        # caption=self.doc['caption']
        #multi caption
        caption_low = self.doc['attributes']['prompt_attr']
        caption_high = self.doc['attributes']['prompt']
        
        print('Generating for ...........', caption_low)

        text_embeddings_low=[]
        prompts_obj_low = []
        prompts_text_low = []
        neg_prompts_obj_low = []
        encodings=self._encode_prompt(
            caption_low,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        negative_prompt_embeds_low, prompt_embeds_low =encodings[0],encodings[1]
        prompts_text_low=caption_low
        prompts_obj_low.append(prompt_embeds_low)
        neg_prompts_obj_low.append(negative_prompt_embeds_low)
        prompts_obj_low = torch.stack(prompts_obj_low, dim=0)
        neg_prompts_obj_low = torch.stack(neg_prompts_obj_low, dim=0)
        if do_classifier_free_guidance:
            text_embeddings_low = torch.cat([neg_prompts_obj_low, prompts_obj_low], dim=0)


        text_embeddings_high=[]
        prompts_obj_high = []
        prompts_text_high = []
        neg_prompts_obj_high = []
        encodings=self._encode_prompt(
            caption_high,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        negative_prompt_embeds_high, prompt_embeds_high =encodings[0],encodings[1]
        prompts_text_high=caption_high
        prompts_obj_high.append(prompt_embeds_high)
        neg_prompts_obj_high.append(negative_prompt_embeds_high)
        prompts_obj_high = torch.stack(prompts_obj_high, dim=0)
        neg_prompts_obj_high = torch.stack(neg_prompts_obj_high, dim=0)
        if do_classifier_free_guidance:
            text_embeddings_high = torch.cat([neg_prompts_obj_high, prompts_obj_high], dim=0)

        # 2. Define call parameters
        if prompts_text_high is not None and isinstance(prompts_text_high, str):
            batch_size = 1
        elif prompts_text_high is not None and isinstance(prompts_text_high, list):
            batch_size = len(prompts_text_high)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        # 5. Prepare latent variables
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompts_obj_high.dtype,
            device,
            generator,
            latents,
        )
        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # NEW - stores the attention calculated in the unet
        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))
        
        self.attention_store = AttentionStore(attn_res)
        self.register_attention_control()
        final_latents = latents.clone().detach()

        self.smoothing = GaussianSmoothing().to(prompts_obj_high.device)
        self.smoothing3d = GaussianSmoothing(channels=attn_res[0]*attn_res[1]).to(prompts_obj_high.device)
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                # NEW
                self.i = i
                if t > 900:
                    prompts_obj = prompts_obj_low
                    text_embeddings = text_embeddings_low
                    prompts_text = prompts_text_low
                else:
                    prompts_obj = prompts_obj_high
                    text_embeddings = text_embeddings_high
                    prompts_text = prompts_text_high
                
                if i < 40:
                    latents, mega_mask = self._step(
                        latents,
                        prompts_obj,
                        t,
                        i,
                        step_size,
                        cross_attention_kwargs,
                        prompts_text,
                        clusters=self.doc['clusters'],
                        grid=self.doc['grid'],
                        objects = self.doc['objects'],
                        quantity_bool=self.doc['quantity']
                    )
                
                latent_model_input = (
                        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    )
                latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    
                    # predict the noise residual
                noise_pred = self.unet(
                        latent_model_input,#[0].unsqueeze(0),
                        t,
                        encoder_hidden_states=text_embeddings,#[0].unsqueeze(0),
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
        
             
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text[0], guidance_rescale=guidance_rescale
                )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            image = final_latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        if not return_dict:
            return (image, has_nsfw_concept)

        return (
            StableDiffusionPipelineOutput(
                images=image, nsfw_content_detected=has_nsfw_concept
            ),
            prompts_text,mega_mask

        )

    def _step(
        self,
        latents,
        text_embeddings,
        t,
        i,
        step_size,
        cross_attention_kwargs,
        prompt,
        clusters,
        grid,
        objects,
        quantity_bool
    ):
        with torch.enable_grad():
            if t>750:
                max_iter_to_alter = 10
            else:
                max_iter_to_alter = 1
            updated_latents = []
            
            for latent, text_embedding in zip(latents, text_embeddings):
                # Forward pass of denoising with text conditioning
                
                text_embedding = text_embedding.unsqueeze(0)
               
                latent = latent[None, ...]
                
                for k in range(max_iter_to_alter):
                    latent = latent.clone().detach().requires_grad_(True)
                    self.unet(
                        latent,
                        t,
                        encoder_hidden_states=text_embedding,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )

                    self.unet.zero_grad()
                    
                    # Get attention maps
                    attention_maps,self_attention_maps = self._aggregate_and_get_attention_maps_per_token()
                    attn_map_idx_to_wp = self.get_attention_map_index_to_wordpiece(
                    self.tokenizer, prompt)
                    
                    obj_list = list(attn_map_idx_to_wp.values())
                    
                    attn_id = {}
                    for key, value in attn_map_idx_to_wp.items():
                        if value in list(attn_id.keys()):
                            attn_id[value].append(key)
                        else:
                            attn_id[value]=[key]
                    
                    if len(clusters)>1:
                        loss,mega_mask = self.multiple_loss(
                    attention_maps=attention_maps, self_attention_maps=self_attention_maps, attn_id= attn_id, positions=clusters, grids=grid,t=t,objects=objects)
                    else:
                        loss,mega_mask = self.single_loss(
                    attention_maps=attention_maps, self_attention_maps=self_attention_maps, attn_id= attn_id, positions=clusters, grids=grid,t=t,objects=objects)
                    # loss1 = self._compute_loss_org(max_attention_per_index)
                    # if t>750:
                    #     loss = 6*loss
                    print(f"Loss: {loss:0.4f}", f"Step: {k}",f"t: {t}")
                    if loss != 0:
                        latent = self._update_latent(
                            latents=latent, loss=loss, step_size=step_size
                        )
                        
                updated_latents.append(latent)
        
        latents = torch.cat(updated_latents, dim=0)

        return latents, mega_mask

    def multiple_loss(self, attention_maps, self_attention_maps, attn_id, positions, grids,t,objects):
        losses = 0
        smooth = 1e-5
      
        masks ={}
        areas = []
        for idx,i in enumerate(positions):
            
            mask = np.zeros(attention_maps[0].shape)
            attr_indexes= torch.zeros_like(attention_maps[0])
            obj_indexes= torch.zeros_like(attention_maps[0])
               
            matching_objects=[]
            matching_attributes=[]
            if i in objects.keys():
                for attr_idx, attr_obj in enumerate(objects[i]):                     
                
                    if attr_obj in attn_id.keys():
                        if attr_idx == 0:
                            matching_objects.append(attr_obj)
                        else:
                            matching_attributes.append(attr_obj)
                    else:
                        if any(sub in attn_id.keys() for sub in attr_obj.split()):
                            sub_string = attr_obj.split()
                            for subs in sub_string:
                                if subs in attn_id.keys():
                                    if attr_idx == 0:
                                        matching_objects.append(subs)
                                    else:
                                        matching_attributes.append(subs)

                        elif any(sub in attn_id.keys() for sub in i.split()):
                            sub_string = i.split()
                            for subs in sub_string:            
                                if subs in attn_id.keys():
                                    if attr_idx == 0:
                                        matching_objects.append(subs)
                                    else:
                                        matching_attributes.append(subs)
                        else:
                            
                            word=''
                            for char in attr_obj:
                                if word in attn_id.keys():
                                    if attr_idx == 0:
                                        matching_objects.append(word)
                                    else:
                                        matching_attributes.append(word)
                                    word = ''
                                else:
                                    word+=char
            
            for each in matching_objects:
                if len(attn_id[each])>0:
                    obj_indexes += attention_maps[attn_id[each][0] if attn_id[each][0]<76 else 76]
                    attn_id[each].pop(0)
                    
                else:
                    continue

            for each in matching_attributes:
                if len(attn_id[each])>0:
                    attr_indexes += attention_maps[attn_id[each][0] if attn_id[each][0]<76 else 76]
                    attn_id[each].pop(0)
                    
                else:
                    continue 
            
            for quant in positions[i]:
                mask = quant[-1]
                mask_inv = 1 - mask
                mask_inv= torch.from_numpy(mask_inv).to(obj_indexes.device)
                center = np.argwhere(mask==1).sum(0)/np.count_nonzero(mask==1)
                
                if len(quant) > 1:

                    y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
                    distance_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
                    max_distance = np.sqrt(center[0]**2 + center[1]**2)
                    mask_1 = (1 - (distance_from_center / max_distance)) * mask
                    
                    mask_2 = distance_transform_edt(mask.astype(np.uint8))
                    mask_2 = cv2.normalize(mask_2, mask_2, 0, 1, cv2.NORM_MINMAX)
                    #new one
                    # mask = distance_transform(mask,grids[positions[i][1]][2])
                    mask = 0.5 * mask_1 + 0.5 * mask_2
                    mask = cv2.normalize(mask, mask, 0, 1, cv2.NORM_MINMAX)
                    # mask = 1-mask
                    
                
                mask = torch.from_numpy(mask).to(obj_indexes.device)
                if len(objects[i]) > 0:
                    masks[objects[i][0]] = mask
                else:
                    pass
                # masks[objects[i][0]] = mask
                
                # if t<750:
                losses += 0.1*self.attr_loss( obj_indexes*(1-mask_inv), attr_indexes*(1-mask_inv))  # Re-enable attr_loss calculation
                attn_indexes = attr_indexes + obj_indexes

                losses += self.dice_loss(attn_indexes, mask, mask_inv) 

                losses += 0.5*self.dice_3d_loss(self_attention_maps, mask, mask_inv)
                area = torch.sum(attn_indexes)
            losses = losses/(len(positions[i]))
            areas.append(area/len(positions[i]))
        
        total_area = torch.sum(torch.ones_like(attn_indexes))  
        losses += 0.0001*self.size_loss(areas,total_area)
        return losses/(len(positions)),masks

    def single_loss(self, attention_maps,self_attention_maps, attn_id, positions, grids,t,objects):
        losses = 0
        smooth = 1e-5
        masks ={}
        for idx,i in enumerate(positions):
            
            mask = np.zeros(attention_maps[0].shape)
            attr_indexes= torch.zeros_like(attention_maps[0])
            obj_indexes= torch.zeros_like(attention_maps[0])
               
            matching_objects=[]
            matching_attributes=[]
            if i in objects.keys():
                for attr_idx, attr_obj in enumerate(objects[i]):                     
                
                    if attr_obj in attn_id.keys():
                        if attr_idx == 0:
                            matching_objects.append(attr_obj)
                        else:
                            matching_attributes.append(attr_obj)
                    else:
                        if any(sub in attn_id.keys() for sub in attr_obj.split()):
                            sub_string = attr_obj.split()
                            for subs in sub_string:
                                if subs in attn_id.keys():
                                    if attr_idx == 0:
                                        matching_objects.append(subs)
                                    else:
                                        matching_attributes.append(subs)

                        elif any(sub in attn_id.keys() for sub in i.split()):
                            sub_string = i.split()
                            for subs in sub_string:            
                                if subs in attn_id.keys():
                                    if attr_idx == 0:
                                        matching_objects.append(subs)
                                    else:
                                        matching_attributes.append(subs)
                        else:
                            
                            word=''
                            for char in attr_obj:
                                if word in attn_id.keys():
                                    if attr_idx == 0:
                                        matching_objects.append(word)
                                    else:
                                        matching_attributes.append(word)
                                    word = ''
                                else:
                                    word+=char
            
            for each in matching_objects:
                if len(attn_id[each])>0:
                    obj_indexes += attention_maps[attn_id[each][0] if attn_id[each][0]<76 else 76]
                    attn_id[each].pop(0)
                    
                else:
                    continue

            for each in matching_attributes:
                if len(attn_id[each])>0:
                    attr_indexes += attention_maps[attn_id[each][0] if attn_id[each][0]<76 else 76]
                    attn_id[each].pop(0)
                else:
                    continue

            for quant in positions[i]:
                mask = quant[-1]
                mask = torch.from_numpy(mask).to(obj_indexes.device)
                masks[objects[i][0]] = mask

            losses += 0.1*self.attr_loss( obj_indexes*mask, attr_indexes*mask)
            attn_indexes = attr_indexes + obj_indexes
            losses += self.dice_loss(attn_indexes, mask)
            losses += 0.5*self.dice_3d_loss(self_attention_maps, mask,1-mask)

        return losses/(len(positions)),masks
            
    def iou_loss(self, attn_indexes, mask, mask_inv=None):
        # loss = 0
        smooth = 1e-5        
        smoothing = GaussianSmoothing().to(attn_indexes.device)
        input_gau = attn_indexes.unsqueeze(0).unsqueeze(0)
        attn_indexes = smoothing(input_gau).squeeze(0).squeeze(0)
        attn_indexes = torch.nn.functional.sigmoid(attn_indexes)
        
        mask= mask.unsqueeze(0).contiguous().view(-1)
        mask_inv= mask_inv.unsqueeze(0).contiguous().view(-1)
        attn_indexes=attn_indexes.unsqueeze(0).contiguous().view(-1)

        intersection = torch.sum(attn_indexes * mask)  # (N, C)
        outersection = torch.sum(attn_indexes * mask_inv)
        coef_in = intersection/(intersection+outersection+smooth)
        coef_out = outersection/(intersection+outersection+smooth)
        # max_out = (attn_indexes * mask_inv).max()

        loss = -(torch.log(coef_in+smooth)+torch.log(1-coef_out+smooth))

        # loss += 0.1*max_out

        return loss

    def attr_loss(self, obj_indexes, attr_indexes):
        bce_loss = torch.nn.BCELoss()
        # loss = mse_loss(obj_indexes, attr_indexes)
        input_gau = attr_indexes.unsqueeze(0).unsqueeze(0)
        attr_indexes = self.smoothing(input_gau).squeeze(0).squeeze(0)
        attr_indexes = torch.nn.functional.sigmoid(attr_indexes)

        input_gau_obj = obj_indexes.unsqueeze(0).unsqueeze(0)
        obj_indexes = self.smoothing(input_gau_obj).squeeze(0).squeeze(0)
        obj_indexes = torch.nn.functional.sigmoid(obj_indexes)

        

        loss = bce_loss(attr_indexes, obj_indexes) + 0.1*self.submask_loss(attr_indexes, obj_indexes)
        return loss

    def submask_loss(self, pred, mask):
        penalty = pred * (1 - mask)
        return penalty.mean()
    
    def get_attention_map_index_to_wordpiece(self, tokenizer, prompt):
        attn_map_idx_to_wp = {}

        wordpieces2indices = self.get_indices(tokenizer, prompt)

        # Ignore `start_token` and `end_token`
        for i in list(wordpieces2indices.keys())[1:-1]:
            wordpiece = wordpieces2indices[i]
            wordpiece = wordpiece.replace("</w>", "")
            attn_map_idx_to_wp[i] = wordpiece

        return attn_map_idx_to_wp
    
    def get_indices(self,tokenizer, prompt: str):
        """Utility function to list the indices of the tokens you wish to alter"""
        ids = tokenizer(prompt).input_ids
        indices = {
            i: tok
            for tok, i in zip(
                tokenizer.convert_ids_to_tokens(ids), range(len(ids))
            )
        }
        return indices

    def dice_loss(self, attn_indexes, mask, mask_inv=None):
        smooth = 1e-5        
        
        input_gau = attn_indexes.unsqueeze(0).unsqueeze(0)
        attn_indexes = self.smoothing(input_gau).squeeze(0).squeeze(0)
        attn_indexes = torch.nn.functional.sigmoid(attn_indexes)
        
        # mask= mask.unsqueeze(0).contiguous()#.view(-1)
        # attn_indexes=attn_indexes.unsqueeze(0).contiguous()#.view(-1)
        intersection = torch.sum(attn_indexes * mask)  # (N, C)
        union = torch.sum(attn_indexes.pow(2)) + torch.sum(mask.pow(2))  # (N, C)

        dice_coef = (2 * intersection + smooth)/ (union + smooth)
        loss = (1 - dice_coef.mean())
        
        if mask_inv is not None:
            #outside loss
            # mask_inv= mask_inv.unsqueeze(0).contiguous().view(-1)
            intersection_out = torch.sum(attn_indexes * mask_inv)  # (N, C)
            union_out = torch.sum(attn_indexes.pow(2)) + torch.sum(mask_inv.pow(2))  # (N, C)
            region_out_sum = torch.sum(attn_indexes.pow(2))
            dice_out_coef = (2 * intersection_out + smooth) / (union_out + smooth)
            loss += dice_out_coef.mean()
        
        return loss

    def dice_3d_loss(self, attn_indexes, mask, mask_inv=None):
        smooth = 1e-5        
        input_gau = attn_indexes.unsqueeze(0)
        
        attn_indexes = self.smoothing3d(input_gau).squeeze(0)
        attn_indexes = torch.nn.functional.sigmoid(attn_indexes)

        # mask= mask.unsqueeze(0).contiguous().view(-1)
        mask_flat = (1-mask_inv).flatten()
        mask_expanded = mask.unsqueeze(0).expand(attn_indexes.shape[0], attn_indexes.shape[1],attn_indexes.shape[1])
        mask_flat_expanded = mask_flat.view(-1, 1, 1).expand(-1, attn_indexes.shape[1], attn_indexes.shape[1])

        mask_final = mask_expanded*mask_flat_expanded

        # attn_indexes = attn_indexes * mask_final
        # attn_indexes = attn_indexes.unsqueeze(0).contiguous().view(-1)

        intersection = torch.sum(attn_indexes * mask_final)  # (N, C)
        union = torch.sum(attn_indexes.pow(2)) + torch.sum(mask_final.pow(2))  # (N, C)
        dice_coef = (2 * intersection + smooth)/ (union + smooth)
        loss = (1 - dice_coef.mean())
        
        # if mask_inv is not None:
        #     #outside loss
        #     mask_inv= mask_inv.unsqueeze(0).contiguous().view(-1)
        #     intersection_out = torch.sum(attn_indexes * mask_inv)  # (N, C)
        #     union_out = torch.sum(attn_indexes.pow(2)) + torch.sum(mask_inv.pow(2))  # (N, C)
        #     region_out_sum = torch.sum(attn_indexes.pow(2))
        #     dice_out_coef = (2 * intersection_out + smooth) / (union_out + smooth)
        #     loss += dice_out_coef.mean()
        
        return loss

    def size_loss(self, attn_area,total_area):
        size_loss = 0
        gemma = 0.0001
        
        for i in range(len(attn_area)-1):
            size_loss += max(0, attn_area[i] - attn_area[i+1] + gemma)
        size_loss = size_loss / total_area
        return size_loss

def _get_attention_maps_list(attention_maps: torch.Tensor) -> List[torch.Tensor]:
    attention_maps *= 100
    
    attention_maps_list = [
        attention_maps[:, :, i] for i in range(attention_maps.shape[2])
    ]
    return attention_maps_list

def is_sublist(sub, main):
    # This function checks if 'sub' is a sublist of 'main'
    return len(sub) < len(main) and all(item in main for item in sub)

def distance_transform(mask,centeroid):
    # mask: numpy array (H, W), centeroid: (y, x)
    # This is already quite efficient: it only computes distances for nonzero pixels.
    # For further optimization, you could use scipy's cdist, but for a single centroid this is not faster.
    y_indices, x_indices = np.nonzero(mask)
    if len(y_indices) == 0:
        return np.zeros_like(mask)
    # Vectorized computation of Euclidean distance from centroid to all mask pixels
    distances = np.sqrt((y_indices - centeroid[0]) ** 2 + (x_indices - centeroid[1]) ** 2)
    distance_map = np.zeros_like(mask, dtype=np.float32)
    distances = cv2.normalize(distances, distances, 0, 1, cv2.NORM_MINMAX)
    distance_map[y_indices, x_indices] = 1-distances
    return distance_map


class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(
            input,
            weight=self.weight.to(input.dtype).to(input.device),
            groups=self.groups, padding='same',
        )