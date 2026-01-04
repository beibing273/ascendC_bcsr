from datasketch import MinHash,MinHashLSH
import scipy.sparse as ssp
import scipy.io as sio
from pathlib import Path
import queue
import time



dirname="/home/user/cudafile/DASP-main/data/dataset"
re_dir_part="reorder_dataset"
re_file_part="_reorder.txt"
re_file_vecpart="_reordervec.txt"
dir_path=Path(dirname)

lsh_threshold=0.3
lsh_num_perm=128
MMA_K=4
def get_re_dir(dir):
    slist=dir.split("/")
    slen=len(slist)
    s1=""
    for i in range(1,slen-1):
        s1=s1+"/"+slist[i]
    return s1+"/"+re_dir_part
#get threshold by average nnz in one column
def get_threshold(avg_col_nnz):
    if(avg_col_nnz<=32):
        return 0.15
    elif(avg_col_nnz<=64):
        return 0.2
    elif(avg_col_nnz<=128):
        return 0.3
    elif(avg_col_nnz<=256):
        return 0.4
    else:
        return 0.5


print("\n*************************** read file and reorder ***************************")
for f in dir_path.glob("small_ww_36_pmec_36.mtx"):
    coo_file=sio.mmread(dirname+"/"+f.name)
    csc_file=coo_file.tocsc()
    rowlen=csc_file.shape[0]
    collen=csc_file.shape[1]
    nnz=csc_file.nnz
    #lsh_threshold=get_threshold(collen/nnz)
    data_list=csc_file.data
    row_ind_list=csc_file.indices
    cols_pointer=csc_file.indptr
    print(f"filename:{f.name}")
    print("file shape:",csc_file.shape)
    print("file nnz:",csc_file.nnz,"\n")


    # vervify_data_list=[[-1 for j in range(collen)] for i in range(rowlen)]
    # for col in range(collen):
    #     start_ind=cols_pointer[col]
    #     end_ind=cols_pointer[col+1]
    #     for ind in range(start_ind,end_ind):
    #       vervify_data_list[row_ind_list[ind]][col]=data_list[ind]
    
    #init
    lsh=MinHashLSH(threshold=lsh_threshold, num_perm=lsh_num_perm)
    zero_que=queue.Queue()
    noen_match_que=queue.Queue()
    mh_dict={}
    reorder_inds=[-1 for i in range(collen)]
    re_order_inds_re=[-1 for i in range(collen)]
    col_flag_list=[1 for i in range(collen)]
    re_col_ind_offset=0

    #taverls the column，compute the 128 minhash in each col,and put it in lash
    t1=time.perf_counter()
    count=0
    temp_offset=0
    for col in range(collen):
        start_ind=cols_pointer[col]
        end_ind=cols_pointer[col+1]
        if end_ind==start_ind:
            zero_que.put(col)
            col_flag_list[col]=0
        mh=MinHash(num_perm=lsh_num_perm)
        for ind in range(start_ind,end_ind):
            mh.update(int(row_ind_list[ind]).to_bytes(4,"big"))
        mh_dict[col]=mh
        lsh.insert(col,mh)
    t2=time.perf_counter()
    print(f"Minhash stage time cost is {(t2-t1):.4f} s")

    #select similar col of each col. if number <4 or empty col,put it at the end of reorderlist
    for col in range(collen):
        if col_flag_list[col]==0:
            continue
        similar_col_list=lsh.query(mh_dict[col])
        if len(similar_col_list)<4:
            for simi_col in similar_col_list:
                if col_flag_list[simi_col]==1:
                    noen_match_que.put(simi_col)
                    col_flag_list[simi_col]=0
                    lsh.remove(simi_col)
        else:
            count=0
            temp_offset=0
            reorder_inds[col]=re_col_ind_offset
            re_order_inds_re[re_col_ind_offset]=col
            re_col_ind_offset+=1
            col_flag_list[col]=0
            lsh.remove(col)
            while count<3 and temp_offset<len(similar_col_list):
               simi_col=similar_col_list[temp_offset]
               if simi_col!=col and col_flag_list[simi_col]==1:
                   reorder_inds[simi_col]=re_col_ind_offset
                   re_order_inds_re[re_col_ind_offset]=simi_col
                   re_col_ind_offset+=1
                   col_flag_list[simi_col]=0
                   lsh.remove(simi_col)
                   count+=1
               temp_offset+=1
    while not noen_match_que.empty():
        temp_col=noen_match_que.get()
        reorder_inds[temp_col]=re_col_ind_offset
        re_order_inds_re[re_col_ind_offset]=temp_col
        re_col_ind_offset+=1
        #col_flag_list[temp_col]=0
        #lsh.remove(temp_col)
    while not zero_que.empty():
        temp_col=zero_que.get()
        reorder_inds[temp_col]=re_col_ind_offset
        re_order_inds_re[re_col_ind_offset]=temp_col
        re_col_ind_offset+=1
        #col_flag_list[temp_col]=0
        #lsh.remove(temp_col)
    t3=time.perf_counter()
    print(f"similar cols finding stage time cost is {(t3-t2):.4f} s\n")

   #verify the correction
    errcount=0
    for i in range(collen):
        if(reorder_inds[i]==-1):
            errcount+=1
    if(errcount==0):
        print("reorder col num is right\n")
    else:
        print("errcount: ",errcount,"\n")
    
    #record the nnz num of new col ind and scan sum
    new_cols_pointer=[0 for i in range(collen+1)]
    for col in range(collen):
        start_ind=cols_pointer[col]
        end_ind=cols_pointer[col+1]
        new_cols_pointer[reorder_inds[col]+1]=end_ind-start_ind
    for col in range(1,collen+1):
        new_cols_pointer[col]+=new_cols_pointer[col-1]
    if(new_cols_pointer[collen]!=nnz):
        print("scan sum is error,scan sum is ",new_cols_pointer[collen])
    
    #put the row index and data into new position
    new_data_list=[0 for i in range(nnz)]
    new_row_ind_list=[-1 for i in range(nnz)]
    for col in range(collen):
        start_ind=cols_pointer[col]
        end_ind=cols_pointer[col+1]
        temp_offset=new_cols_pointer[reorder_inds[col]]
        for i in range(start_ind,end_ind):
            new_data_list[temp_offset]=data_list[i]
            new_row_ind_list[temp_offset]=row_ind_list[i]
            temp_offset+=1
    #transform the format into csr
    csc_temp=ssp.csc_matrix((new_data_list,new_row_ind_list,new_cols_pointer),csc_file.shape)
    csr_file=csc_temp.tocsr()

    # errcount2=0
    # #vercify the correction
    # for row in range(csr_file.shape[1]):
    #     start_ind=csr_file.indptr[row]
    #     end_ind=csr_file.indptr[row+1]
    #     for ind in range(start_ind,end_ind):
    #         if (abs(csr_file.data[ind]-vervify_data_list[row][re_order_inds_re[csr_file.indices[ind]]])>1e-5):
    #             errcount2+=1
    # print("vercify the errcount is",errcount2)

    #file operation
    re_dir=get_re_dir(dirname)
    re_filename=re_dir+"/"+f.stem+re_file_part
    re_filevecname=re_dir+"/"+f.stem+re_file_vecpart
    with open(Path(re_filename),"w",encoding="utf-8") as re_f:
        re_f.write(f"%%prducted by MCRA_spMV,format:csr\n")
        re_f.write(f"{csr_file.shape[0]:d} {csr_file.shape[1]:d} {csr_file.nnz:d}\n")
        temp_pointer=csr_file.indptr
        temp_cols=csr_file.indices
        temp_data=csr_file.data
        for row in range(rowlen+1):
            re_f.write(f"{temp_pointer[row]:d}\n")
        for i in range(nnz):
            re_f.write(f"{temp_cols[i]:d} {temp_data[i]:f}\n")
    with open(Path(re_filevecname),"w",encoding="utf-8") as re_vecf:
        re_vecf.write(f"{collen:d}\n")
        for col in range(collen):
            re_vecf.write(f"{reorder_inds[col]:d}\n")

    

    



