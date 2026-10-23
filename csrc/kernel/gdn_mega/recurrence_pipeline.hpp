#pragma once

// Two FP16 components retain a leading value and its rounding correction.
// Original FP16 inputs multiply both components, accumulating in FP32.
// Recurrent state has an authoritative FP32 copy; inverse and attention
// products use native FP32 operands.
template <int R, int K> struct MegaPair;
#ifdef __CCE_AICORE__
template <int R, int K>
struct MegaPair {
    using Mat = Tile<TileType::Mat, half, R, K, BLayout::ColMajor, R, K, SLayout::RowMajor, 512>;
    Mat high, low;
    AICORE inline void assign(uint32_t address) {
        TASSIGN(high, address); TASSIGN(low, address + R * K * sizeof(half));
    }
};
#endif

// The left operand is an original FP16 input, so its rounding residual is zero.
template<int R,int K,int N,typename Acc,typename A,typename B>
AICORE inline void mega_input_pair_product(Acc& result,A& a,B& b)
{
#if defined(__DAV_CUBE__)
    TileLeft<half,R,K> left;TileRight<half,K,N> high,low;
    TASSIGN(left,0);TASSIGN(high,0);TASSIGN(low,K*N*sizeof(half));
    TEXTRACT(left,a,0,0);TMOV(high,b.high);TMOV(low,b.low);pipe_barrier(PIPE_ALL);
    TMATMUL(result,left,low);pipe_barrier(PIPE_M);
    TMATMUL_ACC(result,result,left,high);pipe_barrier(PIPE_ALL);
#endif
}

#ifdef __CCE_AICORE__
template<int R,int C>
using MegaFloat = Tile<TileType::Mat,float,R,C,BLayout::ColMajor,R,C,SLayout::RowMajor,512>;
#endif

template<int B,int C,typename Mat>
AICORE inline void mega_float_join(Mat& inverse,int offset,__gm__ float* scratch)
{
#if defined(__DAV_CUBE__)
    MegaFloat<B,B> temporary;TASSIGN(temporary,128*1024);
    TileLeft<float,B,B> left;TileRight<float,B,B> right;TileAcc<float,B,B> acc;
    TASSIGN(left,0);TASSIGN(right,0);TASSIGN(acc,0);
    TEXTRACT(left,inverse,offset+B,offset+B);TEXTRACT(right,inverse,offset+B,offset);pipe_barrier(PIPE_ALL);
    TMATMUL(acc,left,right);pipe_barrier(PIPE_ALL);
    GlobalTensor<float,Shape<1,1,1,B,B>,pto::Stride<1,1,1,B,1>> gm(scratch);
    TSTORE(gm,acc);pipe_barrier(PIPE_ALL);TLOAD(temporary,gm);pipe_barrier(PIPE_ALL);
    TMOV(left,temporary);TEXTRACT(right,inverse,offset,offset);pipe_barrier(PIPE_ALL);
    TMATMUL(acc,left,right);pipe_barrier(PIPE_ALL);TSTORE(gm,acc);pipe_barrier(PIPE_ALL);
    Tile<TileType::Mat,float,C,C,BLayout::ColMajor,B,B,SLayout::RowMajor,512> destination;
    TASSIGN(destination,(uint64_t)inverse.data()+(offset*C+(offset+B)*8)*sizeof(float));
    TLOAD(destination,gm);pipe_barrier(PIPE_ALL);
#endif
}

template<int C,typename Mat>
AICORE inline void mega_float_inverse(Mat& inverse,__gm__ float* scratch)
{
#if defined(__DAV_CUBE__)
    for(int off=0;off<C;off+=32)mega_float_join<16,C>(inverse,off,scratch);
    for(int off=0;off<C;off+=64)mega_float_join<32,C>(inverse,off,scratch);
    mega_float_join<64,C>(inverse,0,scratch);
#endif
}

// Assemble two independent left operands in L0A, sharing each R panel in L0B.
// K is split in half so the stacked FP32 operand fits the 64 KiB L0A capacity.
template<int C,int D,typename Acc,typename Attention,typename Gamma,typename Residual>
AICORE inline void mega_combined_residual_products(Acc& result,Attention& attention,Gamma& gamma,Residual& residual)
{
#if defined(__DAV_CUBE__)
    static_assert(C==D);
    TileLeft<float,2*C,C/2> left;
    TileLeft<float,C,C/2> top,bottom;
    TileRight<float,C/2,D> right;
    TASSIGN(left,0);TASSIGN(top,0);TASSIGN(bottom,C*(C/2)*sizeof(float));TASSIGN(right,0);
    for(int part=0;part<2;++part){
        TEXTRACT(top,attention,0,part*(C/2));TEXTRACT(bottom,gamma,0,part*(C/2));
        TEXTRACT(right,residual,part*(C/2),0);pipe_barrier(PIPE_ALL);
        if(part==0)TMATMUL(result,left,right);else TMATMUL_ACC(result,result,left,right);
        pipe_barrier(PIPE_ALL);
    }
#endif
}
