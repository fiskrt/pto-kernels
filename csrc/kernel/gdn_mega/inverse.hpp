#pragma once

// Four independent 16x16 unit-triangular solves share one vector instruction
// stream. The gather repeats each block's elimination coefficient over that
// block's columns. This is forward substitution, never an inverse power series.
#if defined(__DAV_VEC__)
template<typename D,typename S>
AICORE inline void mega_copy_block16(D& dst,S& src) {
    set_vector_mask(0,65535);
    vadds(dst.data(),src.data(),0.0f,16,1,1,D::Cols/8,S::Cols/8);
    PipeBarrierVec();set_vector_mask(-1,-1);
}

template <typename Stripe, typename Identity>
AICORE inline void mega_diagonal_inverses(Stripe& a, Identity& identity, int stripe)
{
    constexpr int B=16, R=64, Base=128*1024, Bytes=B*R*sizeof(float);
    using Batch=Tile<TileType::Vec,float,B,R>;
    using Small=Tile<TileType::Vec,float,B,B>;
    Batch inverse, transposed, temporary, coefficients;
    TASSIGN(inverse,Base); TASSIGN(transposed,Base+Bytes);
    TASSIGN(temporary,Base+2*Bytes); TASSIGN(coefficients,Base+3*Bytes);
    Small block, transpose, scratch;
    TASSIGN(block,Base+4*Bytes); TASSIGN(transpose,Base+4*Bytes+B*B*sizeof(float));
    TASSIGN(scratch,Base+4*Bytes+2*B*B*sizeof(float));
    for (int b=0; b<R/B; ++b) {
        using Source=Tile<TileType::Vec,float,B,128,BLayout::RowMajor,B,B>;
        using Dest=Tile<TileType::Vec,float,B,R,BLayout::RowMajor,B,B>;
        Source src, unit; Dest dst, idst;
        TASSIGN(src,(uint64_t)a.data()+(b*B*128+stripe+b*B)*sizeof(float));
        TASSIGN(unit,(uint64_t)identity.data()+(b*B*128+stripe+b*B)*sizeof(float));
        mega_copy_block16(block,src); PipeBarrierVec(); TTRANS(transpose,block,scratch); PipeBarrierVec();
        TASSIGN(dst,Base+Bytes+b*B*sizeof(float)); TASSIGN(idst,Base+b*B*sizeof(float));
        mega_copy_block16(dst,transpose); mega_copy_block16(idst,unit); PipeBarrierVec();
    }
    Tile<TileType::Vec,uint32_t,1,B*R> indices, unused;
    Tile<TileType::Vec,float,1,B*R> gathered;
    TASSIGN(indices,176*1024); TASSIGN(unused,180*1024);
    TASSIGN(gathered,Base+3*Bytes);
    for (int i=0;i+1<B;++i) {
        Tile<TileType::Vec,float,1,R> solved, source;
        TASSIGN(solved,Base+i*R*sizeof(float));
        TASSIGN(source,Base+Bytes+i*R*sizeof(float));
        TGATHER(gathered,source,indices,unused); PipeBarrierVec();
        TCOLEXPAND(temporary,solved); PipeBarrierVec();
        TMULADDDST(inverse,temporary,coefficients); PipeBarrierVec();
    }
    for (int b=0;b<R/B;++b) {
        using Source=Tile<TileType::Vec,float,B,R,BLayout::RowMajor,B,B>;
        using Dest=Tile<TileType::Vec,float,B,128,BLayout::RowMajor,B,B>;
        Source src; Dest dst;
        TASSIGN(src,Base+b*B*sizeof(float));
        TASSIGN(dst,(uint64_t)a.data()+(b*B*128+stripe+b*B)*sizeof(float));
        mega_copy_block16(dst,src); PipeBarrierVec();
    }
}
#endif
