#pragma once

// Two FP16 components retain a leading value and its rounding correction.
// Products include high*high and both first-order cross corrections, accumulated
// in FP32. The second-order low*low term is omitted; tests enforce the resulting
// error, including near-zero elements. Recurrent state has an FP32 copy.
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

template <int R, int K, int N, bool Accumulate = false, typename Acc, typename A, typename B>
AICORE inline void mega_pair_product(Acc& result, A& a, B& b)
{
#if defined(__DAV_CUBE__)
    TileLeft<half, R, K> ah, al;
    TileRight<half, K, N> bh, bl;
    TASSIGN(ah, 0); TASSIGN(al, R * K * sizeof(half));
    TASSIGN(bh, 0); TASSIGN(bl, K * N * sizeof(half));
    TEXTRACT(ah, a.high, 0, 0); TEXTRACT(al, a.low, 0, 0);
    TMOV(bh, b.high); TMOV(bl, b.low); pipe_barrier(PIPE_ALL);
    if constexpr (Accumulate) TMATMUL_ACC(result, result, ah, bl);
    else TMATMUL(result, ah, bl);
    pipe_barrier(PIPE_M);
    TMATMUL_ACC(result, result, al, bh);pipe_barrier(PIPE_M);
    TMATMUL_ACC(result, result, ah, bh);
    pipe_barrier(PIPE_ALL);
#endif
}

// Obtain X_hi and X_lo directly in L1. Subtracting the exactly representable
// X_hi from its FP32 accumulator preserves the residual before rounding it.
template <int R, int K, typename Acc, typename Identity>
AICORE inline void mega_pair_from_acc(MegaPair<R, K>& pair, Acc& acc, Identity& minus_identity)
{
#if defined(__DAV_CUBE__)
    TMOV(pair.high, acc); pipe_barrier(PIPE_ALL);
    TileLeft<half, R, K> high;
    TileRight<half, K, K> negative;
    TASSIGN(high, 0); TASSIGN(negative, 0);
    TMOV(high, pair.high); TMOV(negative, minus_identity); pipe_barrier(PIPE_ALL);
    TMATMUL_ACC(acc, acc, high, negative); pipe_barrier(PIPE_ALL);
    TMOV(pair.low, acc); pipe_barrier(PIPE_ALL);
#endif
}

// Block-triangular inversion combines already solved 16x16 diagonals.
// The off-diagonal block is still signed A: inv21 = inv22 * A21 * inv11.
template<int B,typename Acc,typename A,typename D>
AICORE inline void mega_pair_slice_product(Acc& acc,A& a,int ar,int ac,D& b,int br,int bc)
{
#if defined(__DAV_CUBE__)
    TileLeft<half,B,B> ah,al;TileRight<half,B,B> bh,bl;
    TASSIGN(ah,0);TASSIGN(al,B*B*sizeof(half));TASSIGN(bh,0);TASSIGN(bl,B*B*sizeof(half));
    TEXTRACT(ah,a.high,ar,ac);TEXTRACT(al,a.low,ar,ac);
    TEXTRACT(bh,b.high,br,bc);TEXTRACT(bl,b.low,br,bc);pipe_barrier(PIPE_ALL);
    TMATMUL(acc,ah,bl);pipe_barrier(PIPE_M);TMATMUL_ACC(acc,acc,al,bh);pipe_barrier(PIPE_M);TMATMUL_ACC(acc,acc,ah,bh);
    pipe_barrier(PIPE_ALL);
#endif
}

template<int B,int C>
AICORE inline void mega_inverse_join(MegaPair<C,C>& inverse,int offset,__gm__ half* negative_gm)
{
#if defined(__DAV_CUBE__)
    MegaPair<B,B> temporary;temporary.assign(128*1024);
    typename MegaPair<B,B>::Mat negative;TASSIGN(negative,192*1024);
    mega_load_packed<B,B,128>(negative,negative,negative_gm,B,B);pipe_barrier(PIPE_ALL);
    TileAcc<float,B,B> acc;TASSIGN(acc,0);
    mega_pair_slice_product<B>(acc,inverse,offset+B,offset+B,inverse,offset+B,offset);
    mega_pair_from_acc<B,B>(temporary,acc,negative);
    mega_pair_slice_product<B>(acc,temporary,0,0,inverse,offset,offset);
    TINSERT(inverse.high,acc,offset+B,offset);pipe_barrier(PIPE_ALL);
    TileLeft<half,B,B> high;TileRight<half,B,B> unit;
    TASSIGN(high,0);TASSIGN(unit,0);
    TEXTRACT(high,inverse.high,offset+B,offset);TMOV(unit,negative);pipe_barrier(PIPE_ALL);
    TMATMUL_ACC(acc,acc,high,unit);pipe_barrier(PIPE_ALL);
    TINSERT(inverse.low,acc,offset+B,offset);pipe_barrier(PIPE_ALL);
#endif
}

template<int C>
AICORE inline void mega_complete_inverse(MegaPair<C,C>& inverse,__gm__ half* negative)
{
#if defined(__DAV_CUBE__)
    static_assert(C==128);
    for(int offset=0;offset<C;offset+=32)mega_inverse_join<16,C>(inverse,offset,negative);
    for(int offset=0;offset<C;offset+=64)mega_inverse_join<32,C>(inverse,offset,negative);
    mega_inverse_join<64,C>(inverse,0,negative);
#endif
}

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
