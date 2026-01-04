import sys
import os
import numpy as np
from datasketch import MinHash,MinHashLSH
import scipy.sparse as ssp
import scipy.io as sio
import queue
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
    if len(header) < 3:
        raise ValueError(f"Invalid header in matrix file: {header}")
    
    # Get dimensions (ignore any additional fields like 'general' or 'symmetric')
    M, K, nnz = map(int, header[:3])
    N = K  # As per problem description
    data_lines = lines[1:]
    
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
        
        print(f"{M} {K} {N} {nnz} {block_rows} 0")
        return
    
    # Parse data lines
    try:
        data = np.array([list(map(float, line.split())) for line in data_lines])
    except ValueError as e:
        raise ValueError(f"Error parsing data lines: {e}. First problematic line: {data_lines[0]}")
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # Convert 1-based indices to 0-based
    rows = data[:, 0].astype(int) - 1
    cols = data[:, 1].astype(int) - 1
    values = data[:, 2].astype(np.float32)
    
    csr_row_ptr,csr_col_idx,csr_vals=coo_to_csr(rows,cols,values,M)
    #new to old ref
    new_csr_row_ptr,new_csr_col_idx,new_csr_vals,reorder_ind_ref=reorder_row_csr_minhash(M,K,nnz,csr_row_ptr.tolist(),csr_col_idx.tolist(),csr_vals.tolist())
    
    
    #csr to coo
    for row in range(M):
        for ind in range(new_csr_row_ptr[row],new_csr_row_ptr[row+1]):
            rows[ind]=row
    cols=np.array(new_csr_col_idx)
    values=np.array(new_csr_vals)


    block_rows = (M + BLOCK_M - 1) // BLOCK_M
    block_cols = (K + BLOCK_K - 1) // BLOCK_K
    blocks = {}


    # Calculate block dimensions
    
    # Dictionary to store blocks: key=(block_row, block_col), value=list of (local_row, local_col, value)
   
    
    # Populate blocks from csr to bcsr
    for r, c, v in zip(rows, cols, values):
        # Skip elements outside matrix dimensions (shouldn't happen, but safe)
        if r >= M or c >= K:
            continue
            
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
    output_dir = os.path.join(os.path.dirname(file_path), sample_name)
    os.makedirs(output_dir, exist_ok=True)
    
    #Save binary files
    row_ptr_np.tofile(os.path.join(output_dir, 'row_ptr.bin'))
    col_idx_np.tofile(os.path.join(output_dir, 'col_idx.bin'))
    values_np.tofile(os.path.join(output_dir, 'values.bin'))
    reorder_ref_np.tofile(os.path.join(output_dir,'reorder_ref.bin'))

    # np.savetxt(os.path.join(output_dir, 'row_ptr.txt'),row_ptr_np,delimiter="\n",fmt="%.10f")
    # np.savetxt(os.path.join(output_dir, 'col_idx.txt'), col_idx_np,delimiter="\n",fmt="%.10f")
    # np.savetxt(os.path.join(output_dir, 'values.txt'),values_np,delimiter="\n",fmt="%.10f")
    # np.savetxt(os.path.join(output_dir, 'reorder_ref.txt'), reorder_ref_np,delimiter="\n",fmt="%d")
    
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
    print(f"{M} {K} {N} {nnz} {block_rows} {len(all_block_cols)}")

def coo_to_csr(rows,cols,values,rowlen):
    csr_row_ptr=np.array([0 for i in range(rowlen+1)])
    csr_col_idx=np.array([-1 for i in range(len(values))])
    csr_vals=np.array([-1 for i in range(len(values))])
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
            print("error:the process of coo transform to csr is error,and the err row is {row}")
    return (csr_row_ptr,csr_col_idx,csr_vals)


def reorder_row_csr_minhash(rowlen,collen,nnz,csr_row_ptr,csr_col_idx,csr_vals,lsh_threshold=0.3,lsh_num_perm=128):
    lsh=MinHashLSH(threshold=lsh_threshold, num_perm=lsh_num_perm)
    zero_que=queue.Queue()
    noen_match_que=queue.Queue()
    mh_dict={}
    reorder_inds=[-1 for i in range(rowlen)]
    reorder_inds_re=[-1 for i in range(rowlen)]
    row_flag_list=[1 for i in range(rowlen)]
    re_row_ind_offset=0

   #record the origin data to compare with reordered data , note that the matrix size must be small
    vervify_csr_val=[[-1 for j in range(collen)] for i in range(rowlen)]
    for row in range(rowlen):
        start_ind=csr_row_ptr[row]
        end_ind=csr_row_ptr[row+1]
        for ind in range(start_ind,end_ind):
          vervify_csr_val[row][csr_col_idx[ind]]=csr_vals[ind]

    count=0
    temp_offset=0
    for row in range(rowlen):
        start_ind=csr_row_ptr[row]
        end_ind=csr_row_ptr[row+1]
        if end_ind==start_ind:
            zero_que.put(row)
            row_flag_list[row]=0
        mh=MinHash(num_perm=lsh_num_perm)
        for ind in range(start_ind,end_ind):
            mh.update(int(csr_col_idx[ind]).to_bytes(4,"big"))
        mh_dict[row]=mh
        lsh.insert(row,mh)

    #select similar row of each row. if number <4 or empty row,put it at the end of reorderlist
    for row in range(rowlen):
        if row_flag_list[row]==0:
            continue
        similar_row_list=lsh.query(mh_dict[row])
        if len(similar_row_list)<4:
            for simi_row in similar_row_list:
                if row_flag_list[simi_row]==1:
                    noen_match_que.put(simi_row)
                    row_flag_list[simi_row]=0
                    lsh.remove(simi_row)
        else:
            count=0
            temp_offset=0
            reorder_inds[row]=re_row_ind_offset
            reorder_inds_re[re_row_ind_offset]=row
            re_row_ind_offset+=1
            row_flag_list[row]=0
            lsh.remove(row)
            while count<3 and temp_offset<len(similar_row_list):
               simi_row=similar_row_list[temp_offset]
               if simi_row!=row and row_flag_list[simi_row]==1:
                   reorder_inds[simi_row]=re_row_ind_offset
                   reorder_inds_re[re_row_ind_offset]=simi_row
                   re_row_ind_offset+=1
                   row_flag_list[simi_row]=0
                   lsh.remove(simi_row)
                   count+=1
               temp_offset+=1
    while not noen_match_que.empty():
        temp_row=noen_match_que.get()
        reorder_inds[temp_row]=re_row_ind_offset
        reorder_inds_re[re_row_ind_offset]=temp_row
        re_row_ind_offset+=1
        #row_flag_list[temp_row]=0
        #lsh.remove(temp_row)
    while not zero_que.empty():
        temp_row=zero_que.get()
        reorder_inds[temp_row]=re_row_ind_offset
        reorder_inds_re[re_row_ind_offset]=temp_row
        re_row_ind_offset+=1
        #row_flag_list[temp_row]=0
        #lsh.remove(temp_row)

   #verify the correction
    errcount=0
    for i in range(rowlen):
        if(reorder_inds[i]==-1):
            errcount+=1
    if(errcount==0):
        print("reorder row num is right\n")
    else:
        print("errcount: ",errcount,"\n")
    
    #record the nnz num of new row ind and scan sum
    new_csr_row_ptr=[0 for i in range(rowlen+1)]
    for row in range(rowlen):
        start_ind=csr_row_ptr[row]
        end_ind=csr_row_ptr[row+1]
        new_csr_row_ptr[reorder_inds[row]+1]=end_ind-start_ind
    for row in range(1,rowlen+1):
        new_csr_row_ptr[row]+=new_csr_row_ptr[row-1]
    if(new_csr_row_ptr[rowlen]!=nnz):
        print("scan sum is error,scan sum is ",new_csr_row_ptr[rowlen])
    
    #put the row index and data into new position
    new_csr_vals=[0 for i in range(nnz)]
    new_col_idx=[-1 for i in range(nnz)]
    for row in range(rowlen):
        start_ind=csr_row_ptr[row]
        end_ind=csr_row_ptr[row+1]
        temp_offset=new_csr_row_ptr[reorder_inds[row]]
        for i in range(start_ind,end_ind):
            new_csr_vals[temp_offset]=csr_vals[i]
            new_col_idx[temp_offset]=csr_col_idx[i]
            temp_offset+=1

    #vercify the correction of reordering , note that the matrix size must be small
    csr_temp=ssp.csr_matrix((new_csr_vals,new_col_idx,new_csr_row_ptr),[rowlen,collen])
    errcount2=0
    for row in range(csr_temp.shape[0]):
        start_ind=csr_temp.indptr[row]
        end_ind=csr_temp.indptr[row+1]
        for ind in range(start_ind,end_ind):
            if (abs(csr_temp.data[ind]-vervify_csr_val[reorder_inds_re[row]][csr_temp.indices[ind]])>1e-5):
                errcount2+=1
    print("vercify the errcount is",errcount2)

    return (new_csr_row_ptr,new_col_idx,new_csr_vals,reorder_inds_re)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_matrix.py <path_to_mtx_file>", file=sys.stderr)
        sys.exit(1)
    
    mtx_file = sys.argv[1]
    parse_mtx_to_bcsr(mtx_file)