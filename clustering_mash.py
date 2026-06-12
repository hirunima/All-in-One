import numpy as np
from scipy.ndimage import gaussian_filter
# Function to create a grid of size (rows x cols)
def create_grid(objects,attn_res):
   
    grids = get_distance(attn_res)
    # print( grids)
    triangle_columns = len(objects)
    assigned_patches = {}
    available_patches = len(list(grids.keys()))
        
    if available_patches < triangle_columns:
        print("Not enough bricks to avoid zero columns.")
        objects = objects[:available_patches]
        triangle_columns = available_patches
    
    # Start with minimal valid triangle shape
    bricks = [i + 1 for i in range(triangle_columns)]
    total = sum(bricks)

    if available_patches == total:
        return bricks

    if available_patches > total:
        # Distribute extras from left to right
        remaining = available_patches - total
        i = 0
        while remaining > 0:
            bricks[i] += 1
            remaining -= 1
            i = (i + 1) % triangle_columns
    else:
        # Remove from the right side (while keeping bricks[i] ≥ 1)
        excess = total - available_patches
        i = triangle_columns - 1
        while excess > 0 and i >= 0:
            remove = min(bricks[i] - 1, excess)
            bricks[i] -= remove
            excess -= remove
            i -= 1
    
    for i in range(triangle_columns):
        assigned_patches[objects[i]] = bricks[i]
    return assigned_patches

def get_distance(attn_res):
   
    if attn_res[0] == 24:
        grids={'top-left':[(0,0),(8,8),(4,4),np.array([[1,0,0],[0,0,0],[0,0,0]])], 'top':[(8,0),(16,8),(12,4),np.array([[0,1,0],[0,0,0],[0,0,0]])], 'top-right':[(16,0),(24,8),(20,4),np.array([[0,0,1],[0,0,0],[0,0,0]])], 'left':[(0,8),(8,16),(4,12),np.array([[0,0,0],[1,0,0],[0,0,0]])], 'center':[(8,8),(16,16),(12,12),np.array([[0,0,0],[0,1,0],[0,0,0]])], 'right':[(16,8),(24,16),(20,12),np.array([[0,0,0],[0,0,1],[0,0,0]])], 'bottom-left':[(0,16),(8,24),(4,20),np.array([[0,0,0],[0,0,0],[1,0,0]])], 'bottom':[(8,16),(16,24),(12,20),np.array([[0,0,0],[0,0,0],[0,1,0]])], 'bottom-right':[(16,16),(24,24),(20,20),np.array([[0,0,0],[0,0,0],[0,0,1]])]}
    elif attn_res[0] == 16:
        grids={'top-left':[(0,0),(4,4),(2,2),np.array([[1,0,0],[0,0,0],[0,0,0]])], 'top':[(4,0),(8,4),(6,2),np.array([[0,1,0],[0,0,0],[0,0,0]])], 'top-right':[(8,0),(12,4),(10,2),np.array([[0,0,1],[0,0,0],[0,0,0]])], 'left':[(0,4),(4,8),(2,6),np.array([[0,0,0],[1,0,0],[0,0,0]])], 'center':[(4,4),(8,8),(6,6),np.array([[0,0,0],[0,1,0],[0,0,0]])], 'right':[(8,4),(12,8),(10,6),np.array([[0,0,0],[0,0,1],[0,0,0]])], 'bottom-left':[(0,8),(4,12),(2,10),np.array([[0,0,0],[0,0,0],[1,0,0]])], 'bottom':[(4,8),(8,12),(6,10),np.array([[0,0,0],[0,0,0],[0,1,0]])], 'bottom-right':[(8,8),(12,12),(10,10),np.array([[0,0,0],[0,0,0],[0,0,1]])]}
    elif attn_res[0] == 128:
        grids={'top-left':[(0,0),(42,42),(21,21),np.array([[1,0,0],[0,0,0],[0,0,0]])], 'top':[(42,0),(84,42),(64,21),np.array([[0,1,0],[0,0,0],[0,0,0]])], 'top-right':[(84,0),(128,42),(106,21),np.array([[0,0,1],[0,0,0],[0,0,0]])], 'left':[(0,42),(42,84),(21,64),np.array([[0,0,0],[1,0,0],[0,0,0]])], 'center':[(42,42),(84,84),(64,64),np.array([[0,0,0],[0,1,0],[0,0,0]])], 'right':[(84,42),(128,84),(106,64),np.array([[0,0,0],[0,0,1],[0,0,0]])], 'bottom-left':[(0,84),(42,128),(21,106),np.array([[0,0,0],[0,0,0],[1,0,0]])], 'bottom':[(42,84),(84,128),(64,106),np.array([[0,0,0],[0,0,0],[0,1,0]])], 'bottom-right':[(84,84),(128,128),(106,106),np.array([[0,0,0],[0,0,0],[0,0,1]])]}

    elif attn_res[0] == 64:
        grids={'top-left':[(0,0),(21,21),(10,10),np.array([[1,0,0],[0,0,0],[0,0,0]])], 'top':[(21,0),(42,21),(32,10),np.array([[0,1,0],[0,0,0],[0,0,0]])], 'top-right':[(42,0),(64,21),(53,10),np.array([[0,0,1],[0,0,0],[0,0,0]])], 'left':[(0,21),(21,42),(10,32),np.array([[0,0,0],[1,0,0],[0,0,0]])], 'center':[(21,21),(42,42),(32,32),np.array([[0,0,0],[0,1,0],[0,0,0]])], 'right':[(42,21),(64,42),(53,32),np.array([[0,0,0],[0,0,1],[0,0,0]])], 'bottom-left':[(0,42),(21,64),(10,53),np.array([[0,0,0],[0,0,0],[1,0,0]])], 'bottom':[(21,42),(42,64),(32,53),np.array([[0,0,0],[0,0,0],[0,1,0]])], 'bottom-right':[(42,42),(64,64),(53,53),np.array([[0,0,0],[0,0,0],[0,0,1]])]}
    
    for key, value in grids.items():
        distance_list = []
        for key_j, value_j in grids.items():
            x3, y3 = value[2]
            x4, y4 = value_j[2]
            distance = np.sqrt((x3 - x4)**2 + (y3 - y4)**2)
            distance_list.append(distance)
        sorted_indices=np.argsort(distance_list)
        sorted_keys = [list(grids.keys())[i] for i in sorted_indices]
        
        grids[key].append(sorted_keys)
    return grids
# Function to cluster grid blocks based on the cluster names list
def cluster_grid_by_names(cluster_names,position,attn_res):
    # Calculate the total number of grid cells
    rows, cols = (3,3)
    grids = get_distance(attn_res)
    
    total_cells = rows * cols
    
    # Number of clusters
    num_clusters = len(cluster_names)
    
    # Calculate the number of cells per cluster (divide as evenly as possible)
    # cells_per_cluster = total_cells // num_clusters
    # remaining_cells = total_cells % num_clusters  # Remainder cells to be distributed

    clusters = {}
    
    for i, cluster_name in enumerate(cluster_names):
        idx = 0
        clusters[cluster_name] = []  # Initialize the cluster
        # Determine how many cells this cluster gets
        try:
            cells_for_this_cluster = cluster_names[cluster_name]  
            position_info = grids[position[cluster_name]]
            grid_positon = position_info[-1].copy()#list(grids.keys())
        except:
            position_info = []
            grid_positon = []
            cells_for_this_cluster = 0
            print(f"Error: {cluster_name} not found in cluster_names or position.")
            pass
            # import ipdb; ipdb.set_trace()
            
        print(cluster_name, 'position_info', position_info)
        # if i < remaining_cells:  # Distribute the remaining cells
        #     cells_for_this_cluster += 1
        # Loop through the grid and assign cells to this cluster
        while len(clusters[cluster_name]) < cells_for_this_cluster:
            # row, col = divmod(idx, cols)  # Get the row and column for this index
             # Get the grid cell position
            
            clusters[cluster_name].append(position_info[-1][idx])  # Add the grid cell to the cluster
            try:
                grid_positon.pop(grid_positon.index(position_info[-1][idx]))
            except:
                import ipdb; ipdb.set_trace()
            idx += 1
        # clusters.append({"cluster_name": cluster_name, "cells": cluster_cells})
    return clusters,grids

def do_binary(positions):
    position_binary = {}
    for pos in positions:

        if positions[pos] == 'top-left':
            position_binary[pos] = '100\n000\n000'
        elif positions[pos] == 'top':
            position_binary[pos] = '010\n000\n000'
        elif positions[pos] == 'top-right':
            position_binary[pos] = '001\n000\n000'
        elif positions[pos] == 'left':
            position_binary[pos] = '000\n100\n000'
        elif positions[pos] == 'center':
            position_binary[pos] = '000\n010\n000'
        elif positions[pos] == 'right':
            position_binary[pos] = '000\n001\n000'
        elif positions[pos] == 'bottom-left':
            position_binary[pos] = '000\n000\n100'
        elif positions[pos] == 'bottom':
            position_binary[pos] = '000\n000\n010'
        elif positions[pos] == 'bottom-right':
            position_binary[pos] = '000\n000\n001'
    return position_binary

def grid_seperations(cluster_names,position_cluster,objects_rel,attn_res):
    grids = get_distance(attn_res)
    clusters_positions = {}
    
    for i, cluster_name in enumerate(cluster_names):
        # if cluster_name.endswith('s'):
        #     single_name = cluster_name[:-1]
        # else:
        #     single_name = cluster_name
        # all_names = list(filter(lambda x: single_name in x, objects_rel))
        
        if len(cluster_name) > 0: 
            
            mask = position_cluster[cluster_name][-1]
            ones_indices = np.argwhere(mask == 1)
    
            
            # Apply KMeans clustering to group the 1s into n clusters
            # kmeans = KMeans(n_clusters=n, random_state=0, n_init='auto')
            # labels = kmeans.fit_predict(ones_indices)
            cluster_centers = []
            step = len(ones_indices) // len(all_names)
            for i in range(len(all_names)):
                try:
                    cluster_centers.append(ones_indices[i * step])
                except IndexError:
                    continue
            
            # Assign each point to the nearest cluster center
            clusters = {i: [] for i in range(len(all_names))}
            for point in ones_indices:
                distances = [np.linalg.norm(point - center) for center in cluster_centers]
                nearest_cluster = np.argmin(distances)
                clusters[nearest_cluster].append(point)
            
            # Create n submasks
            submasks = []
            for cluster_id,names in enumerate(all_names):
                submask = np.zeros_like(mask)
                clusters_positions[names]=position_cluster[cluster_name][:-1]
                for x, y in clusters[cluster_id]:
                    submask[x, y] = 1
                
                clusters_positions[names].append(submask)
        else:
            clusters_positions[names] = position_cluster[cluster_name]
    return clusters_positions

def grid_dual_points(cluster_names,position,objects_rel,attn_res,position_cluster):
    grids = get_distance(attn_res)
    clusters = {}
    
    for i, cluster_name in enumerate(objects_rel):
        
        if len(cluster_name) > 0: 
            # for m in cluster_name:
                
            
            pos = grids[position[cluster_name[0]]][4]
            clusters[cluster_name[0]] = []
            position[cluster_name[0]] = []
            for jdx,j in enumerate(cluster_name):
                clusters[j].append(grids[pos[jdx]][3])
                position[j].append(pos[jdx])
            

            
        else:
            clusters[cluster_name] = [position_cluster[cluster_name]]
            position[cluster_name] = [position[cluster_name]]
    
    return clusters,position, grids

def parse_mask(mask_str):
    return np.array([[int(c) for c in line] for line in mask_str.split('\n')])

def cluster_fuzzy(cluster_names,position):
 
    position_cluster = cluster_names.copy() 
    for i,key in enumerate(cluster_names.keys()):
        cluster_fuzzy = np.ones((3, 3))
        key_pos = position_grid[position[key]]
        for j,key_w in enumerate(cluster_names.keys()):
            if j<i:
                continue
            if key!=key_w:
                grid_side = generate_phi_response(prompt,[key,key_w],type=5)
                # grid_rel = generate_phi_response(prompt,[key,key_w],type=6)
                # subject_ = grid_rel["s"]
                # object_ = grid_rel["o"]
                pos = position_grid[position[key_w]]

                import pdb; pdb.set_trace()   
                if grid_side[0] == 'right':
                    
                    for q in range(0,pos[0]-1):
                        for p in range(3):
                            # if pos[0]-1>=0:
                            
                            print('right',p,q)
                            cluster_fuzzy[p][q] = 0
                elif grid_side[0] == 'left':
                    for q in range(pos[0]+1,3):
                        for p in range(3):
                            print('left',p)
                            cluster_fuzzy[p][pos[0]+1] = 0
                if grid_side[1] == 'above':
                    for q in range(0,pos[1]-1):
                        for p in range(3):
                            # if pos[1]-1>=0:
                            print('above',p,q)
                            cluster_fuzzy[q][p] = 0
                if grid_side[1] == 'below':
                    for q in range(pos[1]+1,3):
                        for p in range(3):
                            print('below',p)
                            cluster_fuzzy[pos[1]+1][p] = 0
        
        cluster_fuzzy[key_pos[1]][key_pos[0]] = 1
        position_cluster[key] = cluster_fuzzy
        import pdb; pdb.set_trace()        

def mask_to_str(mask_array):
    return '\n'.join(''.join(str(int(round(c))) for c in row) for row in mask_array)

def fuzzy_weighted_expand(mask, sigma=1.0):
    mask = mask.astype(float)
    blurred = gaussian_filter(mask, sigma=sigma)
    return blurred

def normalize_masks(masks_dict):
    # Sum all fuzzy masks to detect overlap regions
    total = np.sum(list(masks_dict.values()), axis=0)
    total[total == 0] = 1  # Prevent division by zero
    # Normalize each mask by total to get shared ownership
    normalized = {k: v / total for k, v in masks_dict.items()}
    return normalized

def generate_overlap_aware_masks(input_dict, sigma=1.0):
    fuzzy_masks = {}
    text_masks = add_words_to_mask(input_dict)
    for i,(key, mask_str) in enumerate(input_dict.items()):
        binary_mask = parse_mask(mask_str)
        
        fuzzy_mask = fuzzy_weighted_expand(binary_mask, sigma=sigma[i])
        center = np.array(np.where(fuzzy_mask == 1)).T

        fuzzy_masks[key] = fuzzy_mask
    
    # Resolve overlaps proportionally
    resolved_masks = normalize_masks(fuzzy_masks)
    return fuzzy_masks


