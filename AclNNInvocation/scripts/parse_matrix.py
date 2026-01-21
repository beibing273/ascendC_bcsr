import sys
import os
import numpy as np


def _ceil_to(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base


def parse_mtx_to_bcsr(file_path, BLOCK_M=16, BLOCK_K=16):
    """
    Parses a .mtx file to extract sparse matrix A and convert to BCSR format.

    Additionally (per your requirement):
      - Compute padded dims: M_pad, K_pad
      - Set N_pad = K_pad (so B is square and K/N kept consistent)
      - Generate x2_gm.bin (float16, shape [K_pad, N_pad])
      - Generate golden.bin (float32, shape [M_pad, N_pad])
      - Print padded dims (NOT original dims):
            M_pad K_pad N_pad nnz block_rows block_num
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

    # Original dimensions from MTX (padding BEFORE)
    M, K, nnz = map(int, header[:3])
    data_lines = lines[1:]

    # Compute padded dimensions (padding AFTER)
    block_rows = (M + BLOCK_M - 1) // BLOCK_M
    block_cols = (K + BLOCK_K - 1) // BLOCK_K
    M_pad = block_rows * BLOCK_M
    K_pad = block_cols * BLOCK_K

    # You require B's K and N to be consistent AFTER padding:
    # N_pad == K_pad
    N_pad = 128

    # Output directory: <dir>/<sample_name>/
    sample_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.path.dirname(file_path), sample_name)
    os.makedirs(output_dir, exist_ok=True)

    # Special case: empty matrix
    if nnz == 0 or len(data_lines) == 0:
        row_ptr_np = np.zeros(block_rows + 1, dtype=np.int32)
        col_idx_np = np.array([], dtype=np.int32)
        values_np = np.array([], dtype=np.float16)

        row_ptr_np.tofile(os.path.join(output_dir, 'row_ptr.bin'))
        col_idx_np.tofile(os.path.join(output_dir, 'col_idx.bin'))
        values_np.tofile(os.path.join(output_dir, 'values.bin'))

        # Generate padded B and golden even for empty A (golden will be all zeros)
        # B: only first K rows are meaningful, here K may be 0 allowed; keep deterministic
        rng = np.random.default_rng(abs(hash(sample_name)) % (2**32))
        b_pad = np.zeros((K_pad, N_pad), dtype=np.float16)
        if K > 0:
            b_pad[:K, :N_pad] = rng.integers(1, 11, size=(K, N_pad), dtype=np.int32).astype(np.float16)

        a_pad = np.zeros((M_pad, K_pad), dtype=np.float16)
        golden = (a_pad.astype(np.float32) @ b_pad.astype(np.float32)).astype(np.float32)

        b_pad.tofile(os.path.join(output_dir, 'x2_gm.bin'))
        golden.tofile(os.path.join(output_dir, 'golden.bin'))

        with open(os.path.join(output_dir, 'block_info.txt'), 'w') as f:
            f.write(f"BLOCK_M={BLOCK_M}\n")
            f.write(f"BLOCK_K={BLOCK_K}\n")
            f.write(f"Original_M={M}\n")
            f.write(f"Original_K={K}\n")
            f.write(f"Padded_M={M_pad}\n")
            f.write(f"Padded_K={K_pad}\n")
            f.write(f"Padded_N={N_pad}\n")
            f.write(f"Block_rows={block_rows}\n")
            f.write(f"Block_cols={block_cols}\n")
            f.write(f"Num_blocks=0\n")
            f.write(f"Total_values_stored=0\n")

        # IMPORTANT: print padded dims
        print(f"{M_pad} {K_pad} {N_pad} {nnz} {block_rows} 0")
        return

    # Parse data lines (COO)
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

    # Dictionary to store blocks: key=(block_row, block_col), value=list of (local_row, local_col, value)
    blocks = {}

    # Populate blocks
    for r, c, v in zip(rows, cols, vals):
        if r < 0 or c < 0:
            continue
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
    row_ptr = [0]           # Prefix sum array over block-rows
    all_block_cols = []     # Starting columns (multiple of BLOCK_K) for each stored block
    all_block_vals = []     # Flattened (BLOCK_M*BLOCK_K) values for each stored block

    # Also build dense A_pad for golden (optional but easiest / robust)
    a_pad = np.zeros((M_pad, K_pad), dtype=np.float16)

    # Fill A_pad with original nonzeros
    for r, c, v in zip(rows, cols, vals):
        if 0 <= r < M and 0 <= c < K:
            a_pad[r, c] = np.float16(v)

    # Process each block row
    for br in range(block_rows):
        blocks_in_row = 0
        row_block_cols = []
        row_block_vals = []

        for bc in range(block_cols):
            key = (br, bc)
            if key in blocks:
                blocks_in_row += 1
                row_block_cols.append(bc * BLOCK_K)

                # Create dense 16x16 block with zero padding
                block_data = np.zeros((BLOCK_M, BLOCK_K), dtype=np.float16)

                # Fill non-zero elements
                for lr, lc, v in blocks[key]:
                    gr = br * BLOCK_M + lr
                    gc = bc * BLOCK_K + lc
                    if gr < M and gc < K:
                        block_data[lr, lc] = np.float16(v)

                row_block_vals.append(block_data.flatten())

        row_ptr.append(row_ptr[-1] + blocks_in_row)

        if row_block_cols:
            all_block_cols.extend(row_block_cols)
            all_block_vals.extend(row_block_vals)

    # Convert to numpy arrays
    row_ptr_np = np.array(row_ptr, dtype=np.int32)
    col_idx_np = np.array(all_block_cols, dtype=np.int32)
    values_np = np.concatenate(all_block_vals) if all_block_vals else np.array([], dtype=np.float16)

    # Save BCSR files
    row_ptr_np.tofile(os.path.join(output_dir, 'row_ptr.bin'))
    col_idx_np.tofile(os.path.join(output_dir, 'col_idx.bin'))
    values_np.tofile(os.path.join(output_dir, 'values.bin'))
    #不用时注释掉就可以，只是方便查看重排后的结果
    np.savetxt(os.path.join(output_dir, 'row_ptr.txt'),row_ptr_np,delimiter="\n",fmt="%d")
    np.savetxt(os.path.join(output_dir, 'col_idx.txt'), col_idx_np,delimiter="\n",fmt="%d")
    np.savetxt(os.path.join(output_dir, 'values.txt'),values_np,delimiter="\n",fmt="%.10f")

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
    nozero_rate=round(nnz/(len(all_block_cols)*BLOCK_M*BLOCK_K),2)

    # Save metadata
    with open(os.path.join(output_dir, 'block_info.txt'), 'w') as f:
        f.write(f"BLOCK_M={BLOCK_M}\n")
        f.write(f"BLOCK_K={BLOCK_K}\n")
        f.write(f"Original_M={M}\n")
        f.write(f"Original_K={K}\n")
        f.write(f"Padded_M={M_pad}\n")
        f.write(f"Padded_K={K_pad}\n")
        f.write(f"Padded_N={N_pad}\n")
        f.write(f"Block_rows={block_rows}\n")
        f.write(f"Block_cols={block_cols}\n")
        f.write(f"Num_blocks={len(all_block_cols)}\n")
        f.write(f"Total_values_stored={len(values_np)}\n")

    # IMPORTANT: print padded dims (for your bash script / host to use)
    print(f"{M_pad} {K_pad} {N_pad} {nnz} {block_rows} {len(all_block_cols)} {nozero_rate}")




def parse_mtx_to_bcsr_colcondense(file_path, BLOCK_M=16, BLOCK_K=16):
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
    N = 256  # As per problem description
    data_lines = lines[1:]
    block_rows = (M + BLOCK_M - 1) // BLOCK_M
    block_cols = (K + BLOCK_K - 1) // BLOCK_K
    M_pad=block_rows*BLOCK_M
    K_pad=block_cols*BLOCK_K
    N_pad = 256
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
    
    new_csr_row_ptr, new_csr_col_idx, new_csr_vals = coo_to_csr(rows, cols, vals, M)


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
    for csr_row in range(M):
        if(csr_row%BLOCK_M==0):
            all_col_in_rw=[]
            TCcolcount_rw=0
        for csr_ind in range(new_csr_row_ptr[csr_row],new_csr_row_ptr[csr_row+1]):
            all_col_in_rw.append(new_csr_col_idx[csr_ind])
        if((csr_row%BLOCK_M==BLOCK_M-1 or csr_row==M-1) and (len(all_col_in_rw)!=0)):
            lastcol=-1
            all_col_in_rw.sort()
            rw_now=csr_row//BLOCK_M
            for csr_col in range(len(all_col_in_rw)):
                if lastcol!=all_col_in_rw[csr_col]:
                    lastcol=all_col_in_rw[csr_col]
                    sparseAtoB[rw_partition[rw_now]*BLOCK_K+TCcolcount_rw]=all_col_in_rw[csr_col]
                    unique_col[(rw_now,all_col_in_rw[csr_col])]=TCcolcount_rw
                    TCcolcount_rw+=1
            for zerocol in range(TCcolcount_rw,(TCcolcount_rw+BLOCK_K-1)//BLOCK_K*BLOCK_K):
                sparseAtoB[rw_partition[rw_now]*BLOCK_K+zerocol]=sparseAtoB[rw_partition[rw_now]*BLOCK_K+zerocol-1]
            rw_partition[rw_now+1]=rw_partition[rw_now]+(TCcolcount_rw+BLOCK_K-1)//BLOCK_K
    TCcount=rw_partition[block_rows]
    sparseAtoB=sparseAtoB[0:TCcount*BLOCK_K]
    all_block_vals = [0]*(TCcount*BLOCK_M*BLOCK_K)  # Flattened block values
    
    for csr_row in range(M):
        rw_now=csr_row//BLOCK_M 
        for csr_ind in range(new_csr_row_ptr[csr_row],new_csr_row_ptr[csr_row+1]):
            now_col=unique_col[(rw_now,new_csr_col_idx[csr_ind])]
            TCid=rw_partition[rw_now]+now_col//BLOCK_K
            # if(TCid==rw_partition[block_rows]-1):
            #     print("correct")
            all_block_vals[TCid*BLOCK_M*BLOCK_K+(csr_row%BLOCK_M)*BLOCK_K+now_col%BLOCK_K]=new_csr_vals[csr_ind]



    # # Populate blocks from csr to bcsr
    # for r, c, v in zip(rows_new, cols_new, values_new):
    #     # Skip elements outside matrix dimensions (shouldn't happen, but safe)
    #     if r >= M or c >= K:
    #         continue
    #     # if(r==1):
    #     #     print(c)
    #     block_row = r // BLOCK_M
    #     local_row = r % BLOCK_M
        
    #     if block_row not in blocks:
    #         blocks[block_row] = {}
    #     elif blocks[block_row][c] not in blocks:
    #         blocks[block_row][c]=[]
    #     blocks[block_row][c].append((local_row, v))
    
    # Initialize output arrays
      # Prefix sum array
    #all_block_cols = []  # Starting columns for each block
    

    
    # # Process each block row
    # for br in range(block_rows):
    #     blocks_in_row = 0
    #     row_block_cols = []
    #     row_block_vals = []
        
    #     # Process each block column in this block row
    #     for bc in range(block_cols):
    #         key = (br, bc)
    #         if key in blocks:
    #             blocks_in_row += 1
    #             row_block_cols.append(bc * BLOCK_K)  # Starting column index
                
    #             # Create dense block with zero padding
    #             block_data = np.zeros((BLOCK_M, BLOCK_K), dtype=np.float16)
                
    #             # Fill non-zero elements
    #             for lr, lc, val in blocks[key]:
    #                 # Only fill if within original matrix bounds
    #                 global_row = br * BLOCK_M + lr
    #                 global_col = bc * BLOCK_K + lc
    #                 if global_row < M and global_col < K:
    #                     block_data[lr, lc] = np.float16(val)
                
    #             # Flatten in row-major order
    #             row_block_vals.append(block_data.flatten())
        
    #     # Update prefix sum
    #     row_ptr.append(row_ptr[-1] + blocks_in_row)
    #     # Append row data to global arrays
    #     if row_block_cols:
    #         all_block_cols.extend(row_block_cols)
    #         all_block_vals.extend(row_block_vals)
    
    # Convert to numpy arrays
    rw_ptr_np = np.array(rw_partition, dtype=np.int32)
    TC_col_ref_np = np.array(sparseAtoB, dtype=np.int32)
    values_np = np.array(all_block_vals,dtype=np.float16)

    
    # Create output directory
    sample_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.path.dirname(file_path), sample_name+"_colcondense")
    os.makedirs(output_dir, exist_ok=True)
    
    #Save binary files
    rw_ptr_np.tofile(os.path.join(output_dir, 'rw_ptr.bin'))
    TC_col_ref_np.tofile(os.path.join(output_dir, 'TC_col_ref.bin'))
    values_np.tofile(os.path.join(output_dir, 'values.bin'))
    
    #不用时注释掉就可以，只是方便查看重排后的结果
    np.savetxt(os.path.join(output_dir, 'rw_ptr.txt'),rw_ptr_np,delimiter="\n",fmt="%d")
    np.savetxt(os.path.join(output_dir, 'TC_col_ref.txt'),TC_col_ref_np,delimiter="\n",fmt="%d")
    np.savetxt(os.path.join(output_dir, 'values.txt'),values_np,delimiter="\n",fmt="%.10f")
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
    nozero_rate=round(nnz/(TCcount*BLOCK_M*BLOCK_K),2)
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
    print(f"{M_pad} {K_pad} {N_pad} {nnz} {block_rows} {TCcount} {nozero_rate}")

def coo_to_csr(rows,cols,values,rowlen):
    csr_row_ptr=np.array([0 for i in range(rowlen+1)])
    csr_col_idx=np.array([-1 for i in range(len(values))])
    csr_vals=np.array([-1.0 for i in range(len(values))],np.float16)
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_matrix.py <path_to_mtx_file>", file=sys.stderr)
        sys.exit(1)

    mtx_file = sys.argv[1]
    parse_mtx_to_bcsr_colcondense(mtx_file)