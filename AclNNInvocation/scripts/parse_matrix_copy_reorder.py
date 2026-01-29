import sys
import os
import numpy as np
from datasketch import MinHash,MinHashLSH
import scipy.sparse as ssp
import scipy.io as sio
import queue
import time
def parse_mtx_to_bcsr(file_path, BLOCK_M=16, BLOCK_K=16):
    """
    Parses a .mtx file to extract matrix and convert to BCSR format.
    
    Args:
        file_path (str): The path to the .mtx file.
        BLOCK_M (int): Block size in rows (default 16)
        BLOCK_K (int): Block size in columns (default 16)
    
    Converts the matrix to BCSR format with block size BLOCK_M x BLOCK_K.
    Saves three binary files:
    - row_ptr.bin (int32): Prefix sum of blocks per row window (size BLOCK_M)
    - col_idx.bin (int32): Starting column index for each block (multiple of BLOCK_K)
    - values.bin (float16): All elements in each block (BLOCK_M*BLOCK_K elements per block, row-major)
    """
    # Read file lines and filter comments
    with open(file_path, 'r') as f:
        lines = [line for line in f if not line.startswith('%') and line.strip()]
    
    if not lines:
        raise ValueError("Empty matrix file or only comments found")
    
    # Parse header (first non-comment line)
    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid header in matrix file: {header}")
    
    # Get dimensions (ignore any additional fields like 'general' or 'symmetric')
    M, K, nnz = map(int, header[:3])
    N = 128  # As per problem description
    data_lines = lines[1:]
    block_rows = (M + BLOCK_M - 1) // BLOCK_M
    block_cols = (K + BLOCK_K - 1) // BLOCK_K
    M_pad=block_rows*BLOCK_M
    K_pad=block_cols*BLOCK_K
    N_pad = 128
    blocks = {}
    # Special case: empty matrix
    if nnz == 0 or len(data_lines) == 0:
        block_rows = (M + BLOCK_M - 1) // BLOCK_M
        row_ptr = np.zeros(block_rows + 1, dtype=np.int32)
        col_idx = np.array([], dtype=np.int32)
        values = np.array([], dtype=np.float16)
        
        
        # Save outputs
        sample_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(os.path.dirname(file_path), sample_name)
        os.makedirs(output_dir, exist_ok=True)
        
        row_ptr.tofile(os.path.join(output_dir, 'row_ptr.bin'))
        col_idx.tofile(os.path.join(output_dir, 'col_idx.bin'))
        values.tofile(os.path.join(output_dir, 'values.bin'))
        
        with open(os.path.join(output_dir, 'block_info.txt'), 'w') as f:
            f.write(f"BLOCK_M={BLOCK_M}\n")
            f.write(f"BLOCK_K={BLOCK_K}\n")
            f.write(f"Original_M={M}\n")
            f.write(f"Original_K={K}\n")
            f.write(f"Block_rows={block_rows}\n")
            f.write(f"Block_cols={(K + BLOCK_K - 1) // BLOCK_K}\n")
            f.write(f"Num_blocks=0\n")
            f.write(f"Total_values_stored=0\n")
        
        print(f"{M_pad} {K_pad} {N_pad} {nnz} {block_rows} 0 0")
        return
    
    # Parse data lines
    try:
        data = np.array([list(map(float, line.split())) for line in data_lines])
    except ValueError as e:
        raise ValueError(f"Error parsing data lines: {e}. First problematic line: {data_lines[0]}")
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
  
    if(len(data[0])==3):
      rows = data[:, 0].astype(int) - 1
      cols = data[:, 1].astype(int) - 1
      vals = data[:, 2].astype(np.float16)
    elif(len(data[0])==2):
      rows = data[:, 0].astype(int) - 1
      cols = data[:, 1].astype(int) - 1
      vals = np.ones((nnz),np.float16)
    else:
      print("data format is invalid")

      
    csr_row_ptr, csr_col_idx, csr_vals = coo_to_csr(rows, cols, vals, M)

    # #单重排
    # new_csr_row_ptr, new_csr_col_idx, new_csr_vals, reorder_ind_ref = reorder_row_csr_minhash(
    #     M,K, nnz,BLOCK_K,
    #     csr_row_ptr.tolist(),
    #     csr_col_idx.tolist(),
    #     csr_vals.tolist()
    # )

    # #双重排
    # new_M,new_csr_row_ptr,new_csr_col_idx,new_csr_vals,reorder_ind_ref=reorder_double_row_csr_minhash( M,K, nnz,BLOCK_M,BLOCK_K,csr_row_ptr.tolist(),csr_col_idx.tolist(),csr_vals.tolist())
    # M=new_M
    
    #DTC重排
    new_csr_row_ptr, new_csr_col_idx, new_csr_vals, reorder_ind_ref=reorder_double_DTC(M,BLOCK_M,BLOCK_K,nnz,csr_row_ptr,csr_col_idx,csr_vals)

    block_rows = (M + BLOCK_M - 1) // BLOCK_M
    M_pad=block_rows*BLOCK_M

    #填充
    for padrow in range(M,M_pad):
        reorder_ind_ref.append(padrow)

    new_csr_row_ptr = np.array(new_csr_row_ptr, dtype=np.int32)  
    new_csr_col_idx = np.array(new_csr_col_idx, dtype=np.int32)  
    new_csr_vals = np.array(new_csr_vals, dtype=np.float32)      

    #csr转coo
    rows_new = np.repeat(np.arange(M, dtype=np.int32), np.diff(new_csr_row_ptr))
    cols_new = new_csr_col_idx
    values_new = new_csr_vals
    
    # Dictionary to store blocks: key=(block_row, block_col), value=list of (local_row, local_col, value)
    a_pad = np.zeros((M_pad, K_pad), dtype=np.float16)

    # Fill A_pad with original nonzeros
    for r, c, v in zip(rows_new, cols_new, values_new):
        if 0 <= r < M and 0 <= c < K:
            a_pad[r, c] = np.float16(v)
    # Calculate block dimensions
    
    # Populate blocks from csr to bcsr
    for r, c, v in zip(rows_new, cols_new, values_new):
        # Skip elements outside matrix dimensions (shouldn't happen, but safe)
        if r >= M or c >= K:
            continue
        # if(r==1):
        #     print(c)
        block_row = r // BLOCK_M
        block_col = c // BLOCK_K
        local_row = r % BLOCK_M
        local_col = c % BLOCK_K
        
        key = (block_row, block_col)
        if key not in blocks:
            blocks[key] = []
        blocks[key].append((local_row, local_col, v))
    
    # Initialize output arrays
    row_ptr = [0]  # Prefix sum array
    all_block_cols = []  # Starting columns for each block
    all_block_vals = []  # Flattened block values

    # Process each block row
    for br in range(block_rows):
        blocks_in_row = 0
        row_block_cols = []
        row_block_vals = []
        
        # Process each block column in this block row
        for bc in range(block_cols):
            key = (br, bc)
            if key in blocks:
                blocks_in_row += 1
                row_block_cols.append(bc * BLOCK_K)  # Starting column index
                
                # Create dense block with zero padding
                block_data = np.zeros((BLOCK_M, BLOCK_K), dtype=np.float16)
                
                # Fill non-zero elements
                for lr, lc, val in blocks[key]:
                    # Only fill if within original matrix bounds
                    global_row = br * BLOCK_M + lr
                    global_col = bc * BLOCK_K + lc
                    if global_row < M and global_col < K:
                        block_data[lr, lc] = np.float16(val)
                
                # Flatten in row-major order
                row_block_vals.append(block_data.flatten())
        
        # Update prefix sum
        row_ptr.append(row_ptr[-1] + blocks_in_row)
        # Append row data to global arrays
        if row_block_cols:
            all_block_cols.extend(row_block_cols)
            all_block_vals.extend(row_block_vals)
    
    # Convert to numpy arrays
    row_ptr_np = np.array(row_ptr, dtype=np.int32)
    col_idx_np = np.array(all_block_cols, dtype=np.int32)
    values_np = np.concatenate(all_block_vals) if all_block_vals else np.array([], dtype=np.float16)
    reorder_ref_np=np.array(reorder_ind_ref,dtype=np.int32)
    
    # Create output directory
    sample_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.path.dirname(file_path), sample_name+"_re")
    os.makedirs(output_dir, exist_ok=True)
    
    #Save binary files
    row_ptr_np.tofile(os.path.join(output_dir, 'row_ptr.bin'))
    col_idx_np.tofile(os.path.join(output_dir, 'col_idx.bin'))
    values_np.tofile(os.path.join(output_dir, 'values.bin'))
    reorder_ref_np.tofile(os.path.join(output_dir,'reorder_ref.bin'))


    # #不用时注释掉就可以，只是方便查看重排后的结果
    # np.savetxt(os.path.join(output_dir, 'row_ptr.txt'),row_ptr_np,delimiter="\n",fmt="%d")
    # np.savetxt(os.path.join(output_dir, 'col_idx.txt'), col_idx_np,delimiter="\n",fmt="%d")
    # np.savetxt(os.path.join(output_dir, 'values.txt'),values_np,delimiter="\n",fmt="%.10f")
    # np.savetxt(os.path.join(output_dir, 'reorder_ref.txt'), reorder_ref_np,delimiter="\n",fmt="%d")
    # Generate padded B (x2_gm.bin) with deterministic seed based on sample_name
    rng = np.random.default_rng(abs(hash(sample_name)) % (2**32))
    b_pad = np.zeros((K_pad, N_pad), dtype=np.float16)

    # Fill only the meaningful part (original K rows), rest remains 0
    # This avoids any ambiguity if kernel reads padded rows.
    if K > 0:
        b_pad[:K, :N_pad] = rng.integers(1, 11, size=(K, N_pad), dtype=np.int32).astype(np.float16)

    # Golden: (M_pad x K_pad) @ (K_pad x N_pad) -> (M_pad x N_pad)
    golden = (a_pad.astype(np.float32) @ b_pad.astype(np.float32)).astype(np.float32)

    # Save B and golden
    b_pad.tofile(os.path.join(output_dir, 'x2_gm.bin'))
    golden.tofile(os.path.join(output_dir, 'golden.bin'))
    mean_nnz=round(nnz/(len(all_block_cols)*BLOCK_M*BLOCK_K),2)*BLOCK_M*BLOCK_K
    # Save metadata
    with open(os.path.join(output_dir, 'block_info.txt'), 'w') as f:
        f.write(f"BLOCK_M={BLOCK_M}\n")
        f.write(f"BLOCK_K={BLOCK_K}\n")
        f.write(f"Original_M={M}\n")
        f.write(f"Original_K={K}\n")
        f.write(f"Block_rows={block_rows}\n")
        f.write(f"Block_cols={block_cols}\n")
        f.write(f"Num_blocks={len(all_block_cols)}\n")
        f.write(f"Total_values_stored={len(values_np)}\n")
    
    # Print dimensions for calling script
    print(f"{M_pad} {K_pad} {N_pad} {nnz} {block_rows} {len(all_block_cols)} {mean_nnz}")




def parse_mtx_to_bcsr_colcondense(file_path, BLOCK_M=16, BLOCK_K=16,con_thres=1):
    """
    Parses a .mtx file to extract matrix and convert to BCSR format.
    
    Args:
        file_path (str): The path to the .mtx file.
        BLOCK_M (int): Block size in rows (default 16)
        BLOCK_K (int): Block size in columns (default 16)
    
    Converts the matrix to BCSR format with block size BLOCK_M x BLOCK_K.
    Saves three binary files:
    - row_ptr.bin (int32): Prefix sum of blocks per row window (size BLOCK_M)
    - col_idx.bin (int32): Starting column index for each block (multiple of BLOCK_K)
    - values.bin (float16): All elements in each block (BLOCK_M*BLOCK_K elements per block, row-major)
    """
    # Read file lines and filter comments
    with open(file_path, 'r') as f:
        lines = [line for line in f if not line.startswith('%') and line.strip()]
    
    if not lines:
        raise ValueError("Empty matrix file or only comments found")
    
    # Parse header (first non-comment line)
    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid header in matrix file: {header}")
    
    # Get dimensions (ignore any additional fields like 'general' or 'symmetric')
    M, K, nnz = map(int, header[:3])
    N = 128  # As per problem description
    data_lines = lines[1:]
    block_rows = (M + BLOCK_M - 1) // BLOCK_M
    block_cols = (K + BLOCK_K - 1) // BLOCK_K
    M_pad=block_rows*BLOCK_M
    K_pad=block_cols*BLOCK_K
    N_pad = 128
    blocks = {}
    # Special case: empty matrix
    if nnz == 0 or len(data_lines) == 0:
        block_rows = (M + BLOCK_M - 1) // BLOCK_M
        row_ptr = np.zeros(block_rows + 1, dtype=np.int32)
        col_idx = np.array([], dtype=np.int32)
        values = np.array([], dtype=np.float16)
        
        # Save outputs
        sample_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(os.path.dirname(file_path), sample_name)
        os.makedirs(output_dir, exist_ok=True)
        
        row_ptr.tofile(os.path.join(output_dir, 'row_ptr.bin'))
        col_idx.tofile(os.path.join(output_dir, 'col_idx.bin'))
        values.tofile(os.path.join(output_dir, 'values.bin'))
        
        with open(os.path.join(output_dir, 'block_info.txt'), 'w') as f:
            f.write(f"BLOCK_M={BLOCK_M}\n")
            f.write(f"BLOCK_K={BLOCK_K}\n")
            f.write(f"Original_M={M}\n")
            f.write(f"Original_K={K}\n")
            f.write(f"Block_rows={block_rows}\n")
            f.write(f"Block_cols={(K + BLOCK_K - 1) // BLOCK_K}\n")
            f.write(f"Num_blocks=0\n")
            f.write(f"Total_values_stored=0\n")
        
        print(f"{M_pad} {K_pad} {N_pad} {nnz} {block_rows} 0 0")
        return
    
    # Parse data lines
    try:
        data = np.array([list(map(float, line.split())) for line in data_lines])
    except ValueError as e:
        raise ValueError(f"Error parsing data lines: {e}. First problematic line: {data_lines[0]}")
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    if(len(data[0])==3):
      rows = data[:, 0].astype(int) - 1
      cols = data[:, 1].astype(int) - 1
      vals = data[:, 2].astype(np.float16)
    elif(len(data[0])==2):
      rows = data[:, 0].astype(int) - 1
      cols = data[:, 1].astype(int) - 1
      vals = np.ones((nnz),np.float16)
    else:
      print("data format is invalid")
    
    csr_row_ptr, csr_col_idx, csr_vals = coo_to_csr(rows, cols, vals, M)


    #DTC重排
    new_csr_row_ptr, new_csr_col_idx, new_csr_vals, reorder_ind_ref=reorder_double_DTC(M,BLOCK_M,BLOCK_K,nnz,csr_row_ptr,csr_col_idx,csr_vals)

    block_rows = (M + BLOCK_M - 1) // BLOCK_M
    M_pad=block_rows*BLOCK_M
    #重排序映射添加填充行映射

    #对M_pad进行填充
    for padrow in range(M,M_pad):
        reorder_ind_ref.append(padrow)

    new_csr_row_ptr = np.array(new_csr_row_ptr, dtype=np.int32)  
    new_csr_col_idx = np.array(new_csr_col_idx, dtype=np.int32)  
    new_csr_vals = np.array(new_csr_vals, dtype=np.float32)      

    #csr转coo
    rows_new = np.repeat(np.arange(M, dtype=np.int32), np.diff(new_csr_row_ptr))
    cols_new = new_csr_col_idx
    values_new = new_csr_vals

    

    a_pad = np.zeros((M_pad, K_pad), dtype=np.float16)

    # Fill A_pad with original nonzeros
    for r, c, v in zip(rows_new, cols_new, values_new):
        if 0 <= r < M and 0 <= c < K:
            a_pad[r, c] = np.float16(v)
    # Calculate block dimensions
    
    sparseAtoB=[0]*nnz*BLOCK_K
    # sparseAtoB=[0]*nnz
    rw_partition = [0]*(block_rows+1)
    TCcolcount_rw=0
    TCcolcount=0
    unique_col={}
    BLOCK_K_TILE=BLOCK_K//con_thres
    for csr_row in range(M):
        rw_now=csr_row//BLOCK_M
        if(csr_row%BLOCK_M==0):
            all_col_in_rw=[]
            TCcolcount_rw=0
        for csr_ind in range(new_csr_row_ptr[csr_row],new_csr_row_ptr[csr_row+1]):
            all_col_in_rw.append(new_csr_col_idx[csr_ind]//con_thres)
        if((csr_row%BLOCK_M==BLOCK_M-1 or csr_row==M-1) and (len(all_col_in_rw)!=0)):
            lastcol=-1
            all_col_in_rw.sort()
            for csr_col in range(len(all_col_in_rw)):
                if lastcol!=all_col_in_rw[csr_col]:
                    lastcol=all_col_in_rw[csr_col]
                    sparseAtoB[rw_partition[rw_now]*BLOCK_K_TILE+TCcolcount_rw]=all_col_in_rw[csr_col]
                    unique_col[(rw_now,all_col_in_rw[csr_col])]=TCcolcount_rw
                    TCcolcount_rw+=1
            for zerocol in range(TCcolcount_rw,(TCcolcount_rw+BLOCK_K_TILE-1)//BLOCK_K_TILE*BLOCK_K_TILE):
                sparseAtoB[rw_partition[rw_now]*BLOCK_K_TILE+zerocol]=sparseAtoB[rw_partition[rw_now]*BLOCK_K_TILE+zerocol-1]
        if (csr_row%BLOCK_M==BLOCK_M-1 or csr_row==M-1):
            rw_partition[rw_now+1]=rw_partition[rw_now]+(TCcolcount_rw+BLOCK_K_TILE-1)//BLOCK_K_TILE
    TCcount=rw_partition[block_rows]
    for i in range(0,block_rows):
        if(rw_partition[i+1]<rw_partition[i]):
            print("block_row is {i}")
    #print(f"TCcount is {TCcount}")
    sparseAtoB=sparseAtoB[0:TCcount*BLOCK_K_TILE]
    all_block_vals = [0]*(TCcount*BLOCK_M*BLOCK_K)  # Flattened block values
    
    for csr_row in range(M):
        rw_now=csr_row//BLOCK_M 
        for csr_ind in range(new_csr_row_ptr[csr_row],new_csr_row_ptr[csr_row+1]):
            pre_col=new_csr_col_idx[csr_ind]
            now_col=unique_col[(rw_now,pre_col//con_thres)]*con_thres+pre_col%con_thres
            TCid=rw_partition[rw_now]+now_col//BLOCK_K
            # if(TCid==TCcount-1):
            #     print("correct")
            all_block_vals[TCid*BLOCK_M*BLOCK_K+(csr_row%BLOCK_M)*BLOCK_K+now_col%BLOCK_K]=new_csr_vals[csr_ind]


    # Convert to numpy arrays
    rw_ptr_np = np.array(rw_partition, dtype=np.int32)
    TC_col_ref_np = np.array(sparseAtoB, dtype=np.int32)
    values_np = np.array(all_block_vals,dtype=np.float16)
    reorder_ref_np=np.array(reorder_ind_ref,dtype=np.int32)
    
    # Create output directory
    sample_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.path.dirname(file_path), sample_name+"_re_colcondense")
    os.makedirs(output_dir, exist_ok=True)
    
    #Save binary files
    rw_ptr_np.tofile(os.path.join(output_dir, 'rw_ptr.bin'))
    TC_col_ref_np.tofile(os.path.join(output_dir, 'TC_col_ref.bin'))
    values_np.tofile(os.path.join(output_dir, 'values.bin'))
    reorder_ref_np.tofile(os.path.join(output_dir,'reorder_ref.bin'))
    #不用时注释掉就可以，只是方便查看重排后的结果
    # np.savetxt(os.path.join(output_dir, 'rw_ptr.txt'),rw_ptr_np,delimiter="\n",fmt="%d")
    # np.savetxt(os.path.join(output_dir, 'TC_col_ref.txt'),TC_col_ref_np,delimiter="\n",fmt="%d")
    # np.savetxt(os.path.join(output_dir, 'values.txt'),values_np,delimiter="\n",fmt="%.10f")
    # np.savetxt(os.path.join(output_dir, 'reorder_ref.txt'), reorder_ref_np,delimiter="\n",fmt="%d")
    # Generate padded B (x2_gm.bin) with deterministic seed based on sample_name
    rng = np.random.default_rng(abs(hash(sample_name)) % (2**32))
    b_pad = np.zeros((K_pad, N_pad), dtype=np.float16)

    # Fill only the meaningful part (original K rows), rest remains 0
    # This avoids any ambiguity if kernel reads padded rows.
    if K > 0:
        b_pad[:K, :N_pad] = rng.integers(1, 11, size=(K, N_pad), dtype=np.int32).astype(np.float16)

    # Golden: (M_pad x K_pad) @ (K_pad x N_pad) -> (M_pad x N_pad)
    golden = (a_pad.astype(np.float32) @ b_pad.astype(np.float32)).astype(np.float32)

    # Save B and golden
    b_pad.tofile(os.path.join(output_dir, 'x2_gm.bin'))
    golden.tofile(os.path.join(output_dir, 'golden.bin'))
    mean_nnz=round(nnz/(TCcount*BLOCK_M*BLOCK_K),2)*BLOCK_M*BLOCK_K
    # Save metadata
    with open(os.path.join(output_dir, 'block_info.txt'), 'w') as f:
        f.write(f"BLOCK_M={BLOCK_M}\n")
        f.write(f"BLOCK_K={BLOCK_K}\n")
        f.write(f"Original_M={M}\n")
        f.write(f"Original_K={K}\n")
        f.write(f"Block_rows={block_rows}\n")
        f.write(f"Block_cols={block_cols}\n")
        f.write(f"Num_blocks={TCcount}\n")
        f.write(f"Total_values_stored={len(values_np)}\n")
    
    # Print dimensions for calling script
    print(f"{M_pad} {K_pad} {N_pad} {nnz} {block_rows} {TCcount} {mean_nnz} {con_thres}")

def coo_to_csr(rows,cols,values,rowlen):
    csr_row_ptr=np.array([0 for i in range(rowlen+1)])
    csr_col_idx=np.array([-1 for i in range(len(values))])
    csr_vals=np.array([-1.0 for i in range(len(values))],np.float32)
    for row in rows:
        csr_row_ptr[row+1]+=1
    csr_row_ptr=csr_row_ptr.cumsum()
    row_temp_offset=list(csr_row_ptr)
    for (row,col,value) in zip(rows,cols,values):
        csr_col_idx[row_temp_offset[row]]=col
        csr_vals[row_temp_offset[row]]=value
        row_temp_offset[row]+=1
    #verify the correction
    for row in range(rowlen):
        if(row_temp_offset[row]!=csr_row_ptr[row+1]):
            print("[error]:the process of coo transform to csr is error,and the err row is {row}")
    return (csr_row_ptr,csr_col_idx,csr_vals)



def reorder_row_csr_minhash(rowlen,collen,nnz,block_k,csr_row_ptr,csr_col_idx,csr_vals,lsh_threshold=0.7,lsh_num_perm=128):
    lsh=MinHashLSH(threshold=lsh_threshold, num_perm=lsh_num_perm)
    zero_que,noen_match_que=queue.Queue(),queue.Queue()
    mh_dict={}
    reorder_inds=[-1]*rowlen
    reorder_inds_re=[-1]*rowlen
    row_flag_list=[1]*rowlen
    re_row_ind_offset=0

    complete_block_num=0
    imcomplete_block_num=0

#    #使用小矩阵测试正确性，记录原始格式下的原矩阵
#     vervify_csr_val=[[-1 for j in range(collen)] for i in range(rowlen)]
#     for row in range(rowlen):
#         start_ind=csr_row_ptr[row]
#         end_ind=csr_row_ptr[row+1]
#         for ind in range(start_ind,end_ind):
#           vervify_csr_val[row][csr_col_idx[ind]]=csr_vals[ind]
    #计算所有行哈希，放入lsh桶中
    last_ind=-1
    count=0
    temp_offset=0
    for row in range(rowlen):
        start_ind,end_ind=csr_row_ptr[row],csr_row_ptr[row+1]
        last_ind=-1
        if end_ind==start_ind:
            zero_que.put(row)
            row_flag_list[row]=0
        mh=MinHash(num_perm=lsh_num_perm)
        for ind in range(start_ind,end_ind):
            block_ind=csr_col_idx[ind]//block_k
            if(last_ind!=block_ind):
              mh.update(int(block_ind).to_bytes(4,"big"))
              last_ind=block_ind
        mh_dict[row]=mh
        lsh.insert(row,mh)

    #从lsh中获取当前行的相似行list（注意simi_list中所有simi_row只与当前行相似度确定达到阈值，因此row_window中必须包含当前行），根据list大小确定现在确认新位置还是置后。
    for row in range(rowlen):
        if row_flag_list[row]==0:
            continue
        similar_row_list=lsh.query(mh_dict[row])
        if len(similar_row_list)<8:
            imcomplete_block_num+=1
            for simi_row in similar_row_list:
                if row_flag_list[simi_row]==1:
                    noen_match_que.put(simi_row)
                    row_flag_list[simi_row]=0
                    lsh.remove(simi_row)
        else:
            count=0
            temp_offset=0
            complete_block_num+=1
            reorder_inds[row]=re_row_ind_offset
            reorder_inds_re[re_row_ind_offset]=row
            re_row_ind_offset+=1
            row_flag_list[row]=0
            lsh.remove(row)
            while count<8 and temp_offset<len(similar_row_list):
               simi_row=similar_row_list[temp_offset]
               if simi_row!=row and row_flag_list[simi_row]==1:
                   reorder_inds[simi_row]=re_row_ind_offset
                   reorder_inds_re[re_row_ind_offset]=simi_row
                   re_row_ind_offset+=1
                   row_flag_list[simi_row]=0
                   lsh.remove(simi_row)
                   count+=1
               temp_offset+=1

    for que in [noen_match_que,zero_que]:
      while not que.empty():
          temp_row=que.get()
          reorder_inds[temp_row]=re_row_ind_offset
          reorder_inds_re[re_row_ind_offset]=temp_row
          re_row_ind_offset+=1
          #row_flag_list[temp_row]=0
          #lsh.remove(temp_row)

   #检测是否有行遗漏
    errcount=0
    for i in range(rowlen):
        if(reorder_inds[i]==-1):
            errcount+=1
    if(errcount==0):
        #print("reorder row num is right")
        pass
    else:
        print("reorder errcount: ",errcount)
    #print(f"complete block num is {complete_block_num},and incomplete block num is {imcomplete_block_num}")
    
    #确定排序后的row_ptr
    new_csr_row_ptr=[0]*(rowlen+1)
    for row in range(rowlen):
        start_ind=csr_row_ptr[row]
        end_ind=csr_row_ptr[row+1]
        new_csr_row_ptr[reorder_inds[row]+1]=end_ind-start_ind
    for row in range(rowlen):
        new_csr_row_ptr[row+1]+=new_csr_row_ptr[row]
    if(new_csr_row_ptr[rowlen]!=nnz):
        print("scan sum is error,scan sum is ",new_csr_row_ptr[rowlen])
    
    #col_idx和vals相应的重新排序
    new_csr_vals=[0.0]*nnz
    new_col_idx=[-1]*nnz
    for row in range(rowlen):
        start_ind=csr_row_ptr[row]
        end_ind=csr_row_ptr[row+1]
        temp_offset=new_csr_row_ptr[reorder_inds[row]]
        for i in range(start_ind,end_ind):
            new_csr_vals[temp_offset]=csr_vals[i]
            new_col_idx[temp_offset]=csr_col_idx[i]
            temp_offset+=1

    ##小矩阵检验
    # csr_temp=ssp.csr_matrix((new_csr_vals,new_col_idx,new_csr_row_ptr),[rowlen,collen])
    # errcount2=0
    # for row in range(csr_temp.shape[0]):
    #     if(row<4):
    #        print(f"new row:{row},old row:{reorder_inds_re[row]}:")
    #     start_ind=csr_temp.indptr[row]
    #     end_ind=csr_temp.indptr[row+1]
    #     for ind in range(start_ind,end_ind):
    #         if(row<4):
    #           print(f"col:{csr_temp.indices[ind]:d},val:{csr_temp.data[ind]:f}")
    #         if (abs(csr_temp.data[ind]-vervify_csr_val[reorder_inds_re[row]][csr_temp.indices[ind]])>1e-5):
    #             errcount2+=1
    # print("vercify the errcount is",errcount2)

    return (new_csr_row_ptr,new_col_idx,new_csr_vals,reorder_inds_re)


    
#copy from code of DTC-SPMM and modify, We reuse codes from  (https://github.com/HPMLL/DTC-SpMM_ASPLOS24)
def reorder_double_DTC(num_row,block_m,block_k,nnz,ptr,idx,vals,per=128,thres=0.2,cluster_thres=0.2,cblock_m=128):
    #print("=== Init lsh ===")
    t0 = time.time()
    lsh = MinHashLSH(threshold=thres, num_perm=per)
    allver = []
    lists = [[] for i in range(num_row)]

    for i in range(num_row):
        m = MinHash(num_perm=per)
        lastcol=-1
        for iter in range((int)(ptr[i]), (int)(ptr[i+1])):
            nowblockcol=idx[iter]
            if(lastcol!=nowblockcol):
              m.update(str(nowblockcol).encode('utf-8'))
              lists[i].append(nowblockcol)
              lastcol=nowblockcol
        lsh.insert(i, m)
        allver.append(m)

    t1 = time.time()
    #print("init LSH time (s)", t1 - t0)

    def root(i):
        if cluster_id[i]!=cluster_id[cluster_id[i]]:
            cluster_id[i]=root(cluster_id[i])
        return cluster_id[i]
    #生成独一无二的pair标识
    def makenum(a, b):
        if a > b:
            tmp = a
            a = b
            b = tmp
        return a * num_row + b
    def jd(l1,l2):
        if len(l1) == 0 or len(l2) == 0:
            return 0
        s1 = set(l1)
        s2 = set(l2)
        return (float)(len(s1.intersection(s2))) / len(s1.union(s2))

    class Pair(object):
        def __init__(self,p1,p2,similarity):
            self.p1 = p1
            self.p2 = p2
            self.simi = similarity
        def __lt__(self,other): # operator < 
            return self.simi > other.simi
        def __str__(self):
            return str(self.p1) + ' ' + str(self.p2) + ' ' + str(self.simi)

    que = queue.PriorityQueue()
    sset = set()

    #print("=== DTC reorder===")
    t2 = time.time()
    for i in range(num_row):
        if ptr[i] == ptr[i + 1]:
            continue
        res = lsh.query(allver[i])
        for simi_row in res:
                if simi_row == i or makenum(simi_row, i) in sset:
                    continue
                que.put(Pair(simi_row, i, jd(lists[i],lists[simi_row])))
                sset.add(makenum(i,simi_row))

    #print("queue size: ", que.qsize())
    t3 = time.time()
    #print("query LSH time (s): ", t3 - t2)
    cluster_id = [i for i in range(num_row)]
    cluster_sz = [1 for i in range(num_row)]
    deleted = [0 for i in range(num_row)]
    num_cluster = num_row

    t4 = time.time()
    while (not que.empty()) and num_cluster > 0:
        item = que.get()
        p1 = item.p1
        p2 = item.p2
        sset.remove(makenum(p1, p2))
        if p1 == cluster_id[p1] and p2 == cluster_id[p2]:
            if deleted[p1] or deleted[p2]:
                continue
            if cluster_sz[p1] < cluster_sz[p2]:
                cluster_id[p1] = p2
                num_cluster = num_cluster - 1
                cluster_sz[p2] = cluster_sz[p1] + cluster_sz[p2]
                if cluster_sz[p2] >= block_m:
                    deleted[p2] = 1
                    num_cluster = num_cluster - 1
            else:
                cluster_id[p2] = p1
                num_cluster = num_cluster - 1
                cluster_sz[p1] = cluster_sz[p1] + cluster_sz[p2]
                if cluster_sz[p1] >= block_m:
                    deleted[p1] = 1
                    num_cluster = num_cluster - 1
        else:
            p1 = root(p1)
            p2 = root(p2)
            if deleted[p1] or deleted[p2]:
                continue
            if p1 != p2 and not makenum(p1, p2) in sset:
                que.put(Pair(p1, p2, jd(lists[p1], lists[p2])))
                sset.add(makenum(p1, p2))
    t5 = time.time()
    #print("clustering time (s): ", t5 - t4)

    clusters = {}
    t6 = time.time()
    for i in range(num_row):
        ro = root(i)
        if ro in clusters:
            clusters[ro].append(i)
        else:
            clusters[ro] = [i]

    t7 = time.time()
    #print("put into clusters time (s): ", t7 - t6)
    cluster_num = len(clusters)
    #print("cluster_num:", cluster_num)

    #print("=== Cache-Aware level clustering ===")
    key = list(clusters.keys())
    ## cluster twice to improve cache behaviour
    def makenum_c(a, b):
        if a > b:
            tmp = a
            a = b
            b = tmp
        return a * cluster_num + b

    per_c = 128   # for 4090
    lsh_c = MinHashLSH(threshold=cluster_thres, num_perm=per_c)
    allver_c = []
    lists_c = [[] for i in range(cluster_num)]   # unique column indices for each cluster lists_c[i]: indices for cluster i
    cnt = 0
    for i in clusters:
        m = MinHash(num_perm=per_c)
        list_cluster_i = [] 
        for node in clusters[i]:
            list_cluster_i = list_cluster_i +  lists[node]
        list_cluster_i = list(set(list_cluster_i))
        lists_c[cnt] = list_cluster_i
        for ind in list_cluster_i:
            m.update(str(ind).encode("utf-8"))
        lsh_c.insert(str(cnt), m)
        allver_c.append(m)
        cnt = cnt + 1
    que_c = queue.PriorityQueue()
    sset_c = set()
    t2 = time.time()
    for i in range(cluster_num):
        if i % 1000 == 0:
            #print("reach cluster: ", i)
            pass
        if(len(lists_c[i])==0):
            continue
        res = lsh_c.query(allver_c[i])
        for item in res:
            if (int)(item) == i or makenum_c(i, (int)(item)) in sset_c:
                continue
            if len(lists_c[(int)(item)]) == 0:
                continue
            que_c.put(Pair(i, (int)(item), jd(lists_c[i], lists_c[(int)(item)])))
            sset_c.add(makenum_c(i, (int)(item)))
    #print("cluster queue size:", que_c.qsize())
    t3 = time.time()
    #print("query cluster LSH time (s): ", t3 - t2)
    cluster_id_c = [i for i in range(cluster_num)]
    cluster_sz_c = [1 for i in range(cluster_num)]
    deleted_c = [0 for i in range(cluster_num)]
    num_cluster_c = cluster_num
    # def root_c(i):
    #     while i != cluster_id_c[i]:
    #         cluster_id_c[i] = cluster_id_c[cluster_id_c[i]]
    #         i = cluster_id_c[i]
    #     return i
    def root_c(i):
        if cluster_id_c[i]!=cluster_id_c[cluster_id_c[i]]:
            cluster_id_c[i]=root_c(cluster_id_c[i])
        return cluster_id_c[i]
    t4 = time.time()
    while (not que_c.empty()) and num_cluster_c > 0:
        item = que_c.get()
        p1 = item.p1
        p2 = item.p2
        sset_c.remove(makenum_c(p1, p2))
        if p1 == cluster_id_c[p1] and p2 == cluster_id_c[p2]:
            if deleted_c[p1] or deleted_c[p2]:
                continue
            if cluster_sz_c[p1] < cluster_sz_c[p2]:
                cluster_id_c[p1] = p2
                num_cluster_c = num_cluster_c - 1
                cluster_sz_c[p2] = cluster_sz_c[p1] + cluster_sz_c[p2]
                if cluster_sz_c[p2] >= cblock_m:
                    deleted_c[p2] = 1
                    num_cluster_c = num_cluster_c - 1
            else:
                cluster_id_c[p2] = p1
                num_cluster_c = num_cluster_c - 1
                cluster_sz_c[p1] = cluster_sz_c[p1] + cluster_sz_c[p2]
                if cluster_sz_c[p1] >= cblock_m:
                    deleted_c[p1] = 1
                    num_cluster_c = num_cluster_c - 1
        else:
            p1 = root_c(p1)
            p2 = root_c(p2)
            if deleted_c[p1] or deleted_c[p2]:
                continue
            if p1 != p2 and not makenum_c(p1, p2) in sset_c:
                que_c.put(Pair(p1, p2, jd(lists_c[p1], lists_c[p2])))
                sset_c.add(makenum_c(p1, p2))
    t5 = time.time()
    #print("cluster clustering time (s): ", t5 - t4)
    clusters_c = {}
    t6 = time.time()
    for i in range(cluster_num):
        ro = root_c(i)
        if ro in clusters_c:
            clusters_c[ro].append(i)
        else:
            clusters_c[ro] = [i]
    cluster_cluster_num = len(clusters_c)
    #print("cluster_of_cluster_num: ", cluster_cluster_num)
    t7 = time.time()
    #print("put clusters into clusters time (s): ", t7 - t6)
    # print(clusters_c)

    #print("=== Save results ===")
    reorder_ind_re = []
    for j in clusters_c:
        for k in clusters_c[j]:
            clustersk = clusters[key[k]]
            for item in clustersk:
                reorder_ind_re.append(item)
    
    reorder_ind=[-1]*num_row
    new_ptr=[0]*(num_row+1)
    for i in range(num_row):
        reorder_ind[reorder_ind_re[i]]=i
        new_ptr[i+1]=new_ptr[i]+ptr[reorder_ind_re[i]+1]-ptr[reorder_ind_re[i]]
    for i in range(num_row):
        if reorder_ind[i]==-1:
            print("[error]: DTC_reorder row is left")
    
    new_idx=[0]*nnz
    new_vals=[0.0]*nnz
    for i in range(num_row):
        new_start_ind=new_ptr[i]
        new_end_ind=new_ptr[i+1]
        start_ind=ptr[reorder_ind_re[i]]
        for ind in range(new_start_ind,new_end_ind):
            new_idx[ind]=idx[start_ind]
            new_vals[ind]=vals[start_ind]
            start_ind+=1


    return (new_ptr,new_idx,new_vals,reorder_ind_re)







def reorder_double_row_csr_minhash(rowlen,collen,nnz,block_m,block_k,csr_row_ptr,csr_col_idx,csr_vals,lsh_threshold=0.7,rw_lsh_threshold=0.7,lsh_num_perm=128,padding_threshold=0.75):
    lsh=MinHashLSH(threshold=lsh_threshold, num_perm=lsh_num_perm)
    rw_lsh=MinHashLSH(threshold=rw_lsh_threshold,num_perm=lsh_num_perm)
    zero_que,imcomplete_que=queue.Queue(),queue.Queue()
    mh_dict={}
    rw_mh_dict={}
     
    #新行映射到旧行
    reorder_inds_re=[]

    reorder_inds=[i for i in range(rowlen)]
    row_flag_list=[1]*rowlen
    re_row_ind_offset=0
    rw_dict={}

    complete_block=0
    imcomplete_rows=0

#    #（小矩阵测试重排正确性），记录矩阵原始位置
#     vervify_csr_val=[[-1 for j in range(collen)] for i in range(rowlen)]
#     for row in range(rowlen):
#         start_ind=csr_row_ptr[row]
#         end_ind=csr_row_ptr[row+1]
#         for ind in range(start_ind,end_ind):
#           vervify_csr_val[row][csr_col_idx[ind]]=csr_vals[ind]

    #第一次重排，处理元素是行
    temp_offset=0
    row_count=0
    #lsh获取行哈希值
    for row in range(rowlen):
        start_ind=csr_row_ptr[row]
        end_ind=csr_row_ptr[row+1]
        last_ind=-1
        if end_ind==start_ind:
            zero_que.put(row)
            imcomplete_rows+=1
            row_flag_list[row]=0
            row_count+=1
        mh=MinHash(num_perm=lsh_num_perm)
        for ind in range(start_ind,end_ind):
            block_ind=csr_col_idx[ind]
            if(last_ind!=block_ind):
              mh.update(int(block_ind).to_bytes(4,"big"))
              last_ind=block_ind
        mh_dict[row]=mh
        lsh.insert(row,mh)

    #内部函数，将当前行块中的列合并（列_块序号合并）
    def rw_col_combine(simi_list):
        fi_row=simi_list[0]
        rw_mh=MinHash(num_perm=lsh_num_perm)
        re_set=set()
        for row in simi_list:
            for ind in range(csr_row_ptr[row],csr_row_ptr[row+1]):
                col_block=csr_col_idx[ind]
                if col_block not in re_set:
                    rw_mh.update(int(col_block).to_bytes(4,"big"))
                    re_set.add(col_block)
        rw_mh_dict[fi_row]=rw_mh
        rw_lsh.insert(fi_row,rw_mh)


    window_list=[]
    for row in range(rowlen):
        if row_flag_list[row]==0:
            continue
        similar_row_list=lsh.query(mh_dict[row])
        offset16=0
        window_list.clear()
        row_flag_list[row]=0
        lsh.remove(row)
        window_list.append(row)
        row_count+=1
        while len(window_list)<16 and offset16<len(similar_row_list):
            temp_row=similar_row_list[offset16]
            if row_flag_list[temp_row] and temp_row!=row:
                row_flag_list[temp_row]=0
                lsh.remove(temp_row)
                window_list.append(temp_row)
                row_count+=1
            offset16+=1
        rw_col_combine(window_list)
        rw_dict[row]=[1,len(window_list),list(window_list)]
        # else:        #similar_list len <16
        #     row_flag_list[row]=0
        #     lsh.remove(row)
        #     window_list.append(row)
        #     row_count+=1
        #     for temp_row in similar_row_list:
        #         if row_flag_list[temp_row] and temp_row!=row: 
        #             row_flag_list[temp_row]=0
        #             lsh.remove(temp_row)
        #             window_list.append(temp_row)
        #             row_count+=1
        #     rw_col_combine(window_list)
        #     rw_dict[row]=(1,len(window_list),list(window_list))
    #检验第一次重排得到的行数量
    if(row_count!=rowlen):
        print(f"reorder is error,{abs(row_count-rowlen)} rows are left/overwriten")
    else:
        #print("reorder double first row num is right")
        pass
    
    #第二次重排，处理元素是行块

    #内部函数，高相似度的行块集合凑出近似BLOCK_M大小的大块
    def rw_combine(simi_list,fi_rw):
        if(rw_dict[fi_rw][1]==block_m):
            rw_dict[fi_rw][0]=0
            return ([fi_rw],block_m)
        dp=[[0 for i in range (block_m+1)] for j in range (len(simi_list))]
        re_list=[fi_rw]
        rw_dict[fi_rw][0]=0
        for ind in range(len(simi_list)):
            now_rw=simi_list[ind]
            for ln in range(block_m+1):
                if ind==0:
                    if simi_list[0]==fi_rw or ln<rw_dict[now_rw][1]+rw_dict[fi_rw][1] :
                       dp[0][ln]=rw_dict[fi_rw][1]
                    else:
                       dp[0][ln]=rw_dict[fi_rw][1]+rw_dict[now_rw][1]
                elif now_rw==fi_rw:
                    dp[ind][ln]=dp[ind-1][ln]
                else:
                    if ln>=rw_dict[now_rw][1]+rw_dict[fi_rw][1] and dp[ind-1][ln-rw_dict[now_rw][1]]+rw_dict[now_rw][1]<=block_m:
                        dp[ind][ln]=max(dp[ind-1][ln],dp[ind-1][ln-rw_dict[now_rw][1]]+rw_dict[now_rw][1])
                    else:
                        dp[ind][ln]=dp[ind-1][ln]
        ind=len(simi_list)-1
        tmp_ln=dp[len(simi_list)-1][block_m]
        last_max=dp[ind][tmp_ln]
        ind-=1
        while tmp_ln>rw_dict[fi_rw][1] and ind>=0:
            if dp[ind][tmp_ln]!=last_max:
                tmp_ln-=rw_dict[simi_list[ind+1]][1]
                last_max=dp[ind][tmp_ln]
                re_list.append(simi_list[ind+1])
                rw_dict[simi_list[ind+1]][0]=0
            ind-=1
        if ind==-1 and dp[0][tmp_ln]>rw_dict[fi_rw][1]:
            re_list.append(simi_list[0])
            rw_dict[simi_list[0]][0]=0
            tmp_ln-=rw_dict[simi_list[0]][1]
        #检验
        if tmp_ln!=rw_dict[fi_rw][1]:
            print(f"error,fi_rw is {fi_rw}, fi_ln is {rw_dict[fi_rw][1]} and tmp_ln is {tmp_ln}")
            for i in simi_list:
                print(f"fi is {i}, num is {rw_dict[i][1]}")
            print(f"re_list is {re_list}")
        return re_list,dp[len(simi_list)-1][block_m]


        
    #第二次重排开始，调用内部函数rw_combine,对没有满BLOCK_M的大块采用填充或置后策略
    padding_ind=rowlen
    for fi_rw in rw_dict.keys():
        if(rw_dict[fi_rw][0]==0):
            continue
        rw_simi_list=rw_lsh.query(rw_mh_dict[fi_rw])
        temp_rw_list,temp_rw_len=rw_combine(rw_simi_list,fi_rw)
        #print(f"fi_rw:{fi_rw} , and the temp_rw_len is {temp_rw_len},temp_rw_list is {temp_rw_list}")
        if temp_rw_len==block_m or temp_rw_len>=int(block_m*padding_threshold):
            for temp_fi_rw in temp_rw_list:
                rw_lsh.remove(temp_fi_rw)
                for row in rw_dict[temp_fi_rw][2]:
                    reorder_inds_re.append(row)
                    reorder_inds[row]=re_row_ind_offset
                    re_row_ind_offset+=1
            for zero_row in range(block_m-temp_rw_len):
                reorder_inds_re.append(padding_ind)
                padding_ind+=1
            re_row_ind_offset+=(block_m-temp_rw_len)
            complete_block+=1
        else: 
            for temp_fi_rw in temp_rw_list:
                rw_lsh.remove(temp_fi_rw)
                for row in rw_dict[temp_fi_rw][2]:
                    imcomplete_que.put(row)
                    imcomplete_rows+=1
    for que in [imcomplete_que,zero_que]:
        while not que.empty():
            row=que.get()
            reorder_inds_re.append(row)
            reorder_inds[row]=re_row_ind_offset
            re_row_ind_offset+=1
    new_rowlen=complete_block*block_m+imcomplete_rows
    if(re_row_ind_offset!=new_rowlen):
        print(f"reorder double is error,and the row num is {re_row_ind_offset}")
    if(len(reorder_inds_re)!=re_row_ind_offset or padding_ind!=re_row_ind_offset):
        print(f"the re_indx size is error,and the len of reorder_inds_re is {len(reorder_inds_re)},and the padding ind is {padding_ind}")
    #print(f"new rowlen is {new_rowlen} , complete block is {complete_block} , and total rows of imcomplete block is {imcomplete_rows}")

    #为新row指针赋值，并计算前缀和
    new_csr_row_ptr=[0 for i in range(new_rowlen+1)]
    for row in range(rowlen):
        start_ind=csr_row_ptr[row]
        end_ind=csr_row_ptr[row+1]
        new_csr_row_ptr[reorder_inds[row]+1]=end_ind-start_ind
    for row in range(new_rowlen):
        new_csr_row_ptr[row+1]+=new_csr_row_ptr[row]
    if(new_csr_row_ptr[new_rowlen]!=nnz):
        print("scan sum is error,scan sum is ",new_csr_row_ptr[new_rowlen])
    
    #初始化新的col_idx和vals
    new_csr_vals=[0.0 for i in range(nnz)]
    new_col_idx=[-1 for i in range(nnz)]
    for row in range(rowlen):
        start_ind=csr_row_ptr[row]
        end_ind=csr_row_ptr[row+1]
        temp_offset=new_csr_row_ptr[reorder_inds[row]]
        for i in range(start_ind,end_ind):
            new_csr_vals[temp_offset]=csr_vals[i]
            new_col_idx[temp_offset]=csr_col_idx[i]
            temp_offset+=1
    

    ##（小矩阵测试重排正确性），将重排后的数据进行重映射验证正确性。
    # errcount2=0
    # for row in range(new_rowlen):
    #     startrow=new_csr_row_ptr[row]
    #     endrow=new_csr_row_ptr[row+1]
    #     for ind in range(startrow,endrow):
    #         if(abs(vervify_csr_val[reorder_inds_re[row]][new_col_idx[ind]]-new_csr_vals[ind])>1e-5):
    #             errcount2+=1
    #print("vercify the errcount is",errcount2)

    return (new_rowlen,new_csr_row_ptr,new_col_idx,new_csr_vals,reorder_inds_re)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_matrix.py <path_to_mtx_file>", file=sys.stderr)
        sys.exit(1)
    
    mtx_file = sys.argv[1]
    parse_mtx_to_bcsr_colcondense(mtx_file,con_thres=8)