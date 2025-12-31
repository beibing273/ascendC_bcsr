import sys
import os
import numpy as np

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
    
    # Calculate block dimensions
    block_rows = (M + BLOCK_M - 1) // BLOCK_M
    block_cols = (K + BLOCK_K - 1) // BLOCK_K
    
    # Dictionary to store blocks: key=(block_row, block_col), value=list of (local_row, local_col, value)
    blocks = {}
    
    # Populate blocks
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
    
    # Create output directory
    sample_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.path.dirname(file_path), sample_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save binary files
    row_ptr_np.tofile(os.path.join(output_dir, 'row_ptr.bin'))
    col_idx_np.tofile(os.path.join(output_dir, 'col_idx.bin'))
    values_np.tofile(os.path.join(output_dir, 'values.bin'))
    
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
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_matrix.py <path_to_mtx_file>", file=sys.stderr)
        sys.exit(1)
    
    mtx_file = sys.argv[1]
    parse_mtx_to_bcsr(mtx_file)