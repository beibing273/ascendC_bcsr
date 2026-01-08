#ifndef RE_HANDLE
#define RE_HANDLE
#include"common.h"
#include<cstring>
#include <cstdint>
using namespace std;




template<typename T,int64_t M=10,int64_t N=10>
class ReHandler{
/*
   专门用于处理spmv计算结果的类，若需要什么处理逻辑，请向其中添加函数
*/
  private:
   T* re;
   int64_t m;
   int64_t n;
  public:
ReHandler(){
   m=M;
   n=N;
   re=new T[m*n];
}

ReHandler(const T* output,int64_t o_len,int64_t o_m,int64_t o_n){
   if(o_len!=o_m*o_n){
      ERROR_LOG("output size is error!\n");
      re=nullptr;
      return;
   }
   m=o_m;
   n=o_n;
   re=new T[m*n];
   memcpy(re,output,m*n*sizeof(T));
}


~ReHandler(){
   delete[] re;
}

void *get_data(){
   return (void*)re;
}

void* reorder_and_get(int64_t* reorder_ref,int64_t ref_m){
  if(m!=ref_m){
   ERROR_LOG("reorder_ref isn't match to m\n");
   return nullptr;
  }
  T* temp=new T[m*n];
  for(int i=0;i<m;++i){
    cout<<"new row is "<<i<<" and old row is "<<reorder_ref[i]<<std::endl;
     for(int j=0;j<n;++j){
      temp[i*n+j]=re[reorder_ref[i]*n+j];
     }
  }
  T* temp2=re;
  re=temp;
  delete[] temp2;
  return (void*)re;
}
};

#endif