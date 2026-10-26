#pragma once

// exp(x)-1 without cancellation near zero. For |x| <= 1/8 the omitted
// term is below 1.5e-12; FP32 evaluation error dominates that bound.
AICORE inline float mega_small_expm1(float x)
{
    return x*(1.0f+x*(0.5f+x*(1.0f/6.0f+x*(1.0f/24.0f+
        x*(1.0f/120.0f+x*(1.0f/720.0f+x*(1.0f/5040.0f)))))));
}

// Public scratch allocations retain their original sizes. Paired operands use
// separate existing half buffers; the recurrent FP32 state stays in vector UB.
template <int R,int K,typename T,typename Full>
AICORE inline void mega_vector_load(Full& full,__gm__ T* pointer,int rows,int cols,int stride)
{
#if defined(__DAV_VEC__)
    using G=GlobalTensor<T,Shape<1,1,1,DYNAMIC,DYNAMIC>,pto::Stride<1,1,1,DYNAMIC,1>>;
    pipe_barrier(PIPE_ALL);
    if(rows==R && cols==K){G gm(pointer,{1,1,1,R,K},{1,1,1,stride,1});TLOAD(full,gm);pipe_barrier(PIPE_ALL);return;}
    TEXPANDS(full,static_cast<T>(0));PipeBarrierVec();
    if(rows>0){
        using Part=Tile<TileType::Vec,T,R,K,BLayout::RowMajor,DYNAMIC,DYNAMIC,SLayout::NoneBox,512,PadValue::Zero>;
        Part part(rows,cols);TASSIGN(part,(uint64_t)full.data());
        G gm(pointer,{1,1,1,rows,cols},{1,1,1,stride,1});
        pipe_barrier(PIPE_ALL);TLOAD(part,gm);pipe_barrier(PIPE_ALL);
        Tile<TileType::Vec,T,R,K,BLayout::RowMajor,R,K,SLayout::NoneBox,512,PadValue::Zero> padded;
        TASSIGN(padded,(uint64_t)full.data());TFILLPAD_INPLACE(padded,part);PipeBarrierVec();
    }
#endif
}

template<int R,int K,typename F,typename Half>
AICORE inline void mega_split_pair(F& value,Half& high,Half& low)
{
#if defined(__DAV_VEC__)
    constexpr int Bytes=R*K*sizeof(float),Base=96*1024;
    Tile<TileType::Vec,float,R,K> rounded;
    TASSIGN(rounded,Base);TASSIGN(high,Base+Bytes);TASSIGN(low,Base+Bytes+Bytes/2);
    TCVT(high,value,RoundMode::CAST_RINT);PipeBarrierVec();
    TCVT(rounded,high,RoundMode::CAST_NONE);PipeBarrierVec();
    TSUB(rounded,value,rounded);PipeBarrierVec();
    TCVT(low,rounded,RoundMode::CAST_RINT);pipe_barrier(PIPE_ALL);
#endif
}

template<int R,int K,typename F>
AICORE inline void mega_store_pair(F& value,__gm__ half* high_gm,__gm__ half* low_gm,
                                   int rows=R,int stride=K)
{
#if defined(__DAV_VEC__)
    Tile<TileType::Vec,half,R,K> high,low;
    mega_split_pair<R,K>(value,high,low);
    if(rows>0){
        using Part=Tile<TileType::Vec,half,R,K,BLayout::RowMajor,DYNAMIC,K>;
        Part ht(rows),lt(rows);TASSIGN(ht,(uint64_t)high.data());TASSIGN(lt,(uint64_t)low.data());
        using G=GlobalTensor<half,Shape<1,1,1,DYNAMIC,K>,pto::Stride<1,1,1,DYNAMIC,1>>;
        G hg(high_gm,{1,1,1,rows,K},{1,1,1,stride,1}),lg(low_gm,{1,1,1,rows,K},{1,1,1,stride,1});
        TSTORE(hg,ht);TSTORE(lg,lt);pipe_barrier(PIPE_ALL);
    }
#endif
}

template<int R,int K>
AICORE inline void mega_load_pair(MegaPair<R,K>& pair,__gm__ half* high,__gm__ half* low,
                                   int rows=R,int stride=K)
{
#if defined(__DAV_CUBE__)
    using Part=Tile<TileType::Mat,half,R,K,BLayout::ColMajor,DYNAMIC,K,SLayout::RowMajor,512,PadValue::Zero>;
    Part ht(rows),lt(rows);TASSIGN(ht,(uint64_t)pair.high.data());TASSIGN(lt,(uint64_t)pair.low.data());
    using G=GlobalTensor<half,Shape<1,1,1,DYNAMIC,K>,pto::Stride<1,1,1,DYNAMIC,1>>;
    G hg(high,{1,1,1,rows,K},{1,1,1,stride,1}),lg(low,{1,1,1,rows,K},{1,1,1,stride,1});
    TLOAD(ht,hg);TLOAD(lt,lg);pipe_barrier(PIPE_ALL);
    if(rows!=R){TFILLPAD(ht,ht);TFILLPAD(lt,lt);pipe_barrier(PIPE_ALL);}
#endif
}

// FP32 dot products are split by columns across two existing scratch buffers,
// not rounded to half. A 64x128 stripe can reconstruct them without allocation.
template<int R,int C,typename F>
AICORE inline void mega_load_kk(F& value,__gm__ float* first,__gm__ float* second,int rows,int stride)
{
#if defined(__DAV_VEC__)
    using Part=Tile<TileType::Vec,float,R,C,BLayout::RowMajor,DYNAMIC,DYNAMIC,SLayout::NoneBox,512,PadValue::Zero>;
    TEXPANDS(value,0.0f);PipeBarrierVec();
    if(rows>0){
        Part a(rows,C/2),b(rows,C/2);TASSIGN(a,(uint64_t)value.data());TASSIGN(b,(uint64_t)value.data()+C/2*sizeof(float));
        using G=GlobalTensor<float,Shape<1,1,1,DYNAMIC,C/2>,pto::Stride<1,1,1,DYNAMIC,1>>;
        G ag(first,{1,1,1,rows,C/2},{1,1,1,stride,1}),bg(second,{1,1,1,rows,C/2},{1,1,1,stride,1});
        pipe_barrier(PIPE_ALL);TLOAD(a,ag);TLOAD(b,bg);pipe_barrier(PIPE_ALL);
        Part complete(rows,C);TASSIGN(complete,(uint64_t)value.data());
        Tile<TileType::Vec,float,R,C,BLayout::RowMajor,R,C,SLayout::NoneBox,512,PadValue::Zero> padded;
        TASSIGN(padded,(uint64_t)value.data());TFILLPAD_INPLACE(padded,complete);PipeBarrierVec();
    }
#endif
}

// Keep [32,96) KiB untouched: output base and recurrent state live there.
// Weight the next chunk before waiting for current projections. Keep its
// pending coefficients at [96,128) KiB across RHS preparation, then finish
// and export its inverse only after the cube has consumed the current one.
template<int H,int C>
AICORE inline bool mega_prepare_weights(__gm__ float* g,__gm__ float* glow,__gm__ half* beta,__gm__ float* kk,__gm__ float* kk_upper,
    __gm__ float* qk,__gm__ float* lower,__gm__ float* full,__gm__ half* attention_high,
    __gm__ half* attention_low,int64_t pos,int valid,int head,int kh,int64_t tokens)
{
#if defined(__DAV_VEC__)
    constexpr int R=C/2,Gate=160*1024;
    int stripe=get_subblockid()*R,count=max(0,min(R,valid-stripe)),key_head=head/(H/kh);
    using M=Tile<TileType::Vec,float,R,C>;
    using G=Tile<TileType::Vec,float,1,C>;
    using Col=Tile<TileType::Vec,float,R,1,BLayout::ColMajor,R,1>;
    G gates,lows,betas;Tile<TileType::Vec,half,1,C> beta_h;
    TASSIGN(gates,Gate);TASSIGN(lows,Gate+512);TASSIGN(betas,Gate+1024);TASSIGN(beta_h,Gate+1536);
    mega_vector_load<1,C>(gates,g+head*tokens+pos,1,valid,C);
    mega_vector_load<1,C>(lows,glow+head*tokens+pos,1,valid,C);
    mega_vector_load<1,C>(beta_h,beta+head*tokens+pos,1,valid,C);
    TCVT(betas,beta_h,RoundMode::CAST_NONE);PipeBarrierVec();
    M a,raw,mask;TASSIGN(a,96*1024);TASSIGN(raw,0);TASSIGN(mask,128*1024);
    Col gcol,lcol,bcol;
    TASSIGN(gcol,Gate+stripe*sizeof(float));TASSIGN(lcol,Gate+512+stripe*sizeof(float));
    TASSIGN(bcol,Gate+1024+stripe*sizeof(float));
    // One decay matrix serves both QK and KK. Native FP32 attention stores
    // leave this tile intact, unlike the former paired-operand conversion.
    TCOLEXPAND(raw,gates);PipeBarrierVec();TMULS(raw,raw,-1.0f);PipeBarrierVec();
    TROWEXPANDADD(raw,raw,gcol);PipeBarrierVec();
    TCOLEXPAND(a,lows);PipeBarrierVec();TMULS(a,a,-1.0f);PipeBarrierVec();
    TROWEXPANDADD(a,a,lcol);PipeBarrierVec();TADD(raw,raw,a);PipeBarrierVec();
    // Invalid rows must be masked before exp: a padded zero prefix minus
    // a large negative real prefix would otherwise overflow, then 0*inf
    // would poison the solve and even valid output rows.
    mega_vector_load<R,C>(a,full+stripe*C,count,C,C);
    TMUL(raw,raw,a);PipeBarrierVec();TEXP(raw,raw);PipeBarrierVec();TMUL(raw,raw,a);pipe_barrier(PIPE_ALL);
    mega_vector_load<R,C>(a,qk+((pos+stripe)*kh+key_head)*C,count,C,kh*C);
    TMUL(a,a,raw);pipe_barrier(PIPE_ALL);
    if(count>0){
            auto storage=reinterpret_cast<__gm__ float*>(head<H/2?attention_high:attention_low);
            GlobalTensor<float,Shape<1,1,1,DYNAMIC,C>,pto::Stride<1,1,1,C,1>> gm(
                storage+((head%(H/2))*tokens+pos+stripe)*C,{1,1,1,count,C});
            Tile<TileType::Vec,float,R,C,BLayout::RowMajor,DYNAMIC,C> rows(count);TASSIGN(rows,96*1024);
            TSTORE(gm,rows);pipe_barrier(PIPE_ALL);
    }
    mega_load_kk<R,C>(a,kk+((pos+stripe)*kh+key_head)*(C/2),
                       kk_upper+((pos+stripe)*kh+key_head)*(C/2),count,kh*C/2);
    TMUL(a,a,raw);PipeBarrierVec();
    mega_vector_load<R,C>(mask,lower+stripe*C,R,C,C);
    TMUL(a,a,mask);PipeBarrierVec();TROWEXPANDMUL(a,a,bcol);PipeBarrierVec();TMULS(a,a,-1.0f);PipeBarrierVec();
    TABS(raw,a);PipeBarrierVec();
    Tile<TileType::Vec,float,1,R*C> flattened;TASSIGN(flattened,0);
    Tile<TileType::Vec,float,1,8> maximum;
    Tile<TileType::Vec,float,1,1024> reduction;
    TASSIGN(maximum,170*1024);TASSIGN(reduction,171*1024);
    TROWMAX(maximum,flattened,reduction);pipe_barrier(PIPE_ALL);
    bool strongly_coupled=maximum.GetValue(0)>0.5f;
    return strongly_coupled;
#else
    return false;
#endif
}

template<int H,int C>
AICORE inline void mega_finish_inverse(__gm__ float* g,__gm__ float* glow,__gm__ float* lower,__gm__ float* full,__gm__ half* weighted_first,
    int64_t pos,int valid,int head,int kh,int64_t tokens,__gm__ half* first,__gm__ half* second,
    __gm__ half* original_k,__gm__ half* weighted_second,__gm__ half* control,bool strongly_coupled)
{
#if defined(__DAV_VEC__)
    constexpr int R=C/2,Gate=160*1024;
    int stripe=get_subblockid()*R,count=max(0,min(R,valid-stripe)),key_head=head/(H/kh);
    using M=Tile<TileType::Vec,float,R,C>;
    using G=Tile<TileType::Vec,float,1,C>;
    using Col=Tile<TileType::Vec,float,R,1,BLayout::ColMajor,R,1>;
    M a;TASSIGN(a,96*1024);
    G gates,lows;TASSIGN(gates,Gate);TASSIGN(lows,Gate+512);
    mega_vector_load<1,C>(gates,g+head*tokens+pos,1,valid,C);
    mega_vector_load<1,C>(lows,glow+head*tokens+pos,1,valid,C);
    M raw,mask;TASSIGN(raw,0);TASSIGN(mask,128*1024);
    mega_vector_load<R,C>(raw,full+stripe*C,R,C,C);
    mega_vector_load<R,C>(mask,lower+stripe*C,R,C,C);TSUB(raw,raw,mask);PipeBarrierVec();
    mega_diagonal_inverses(a,raw,stripe);pipe_barrier(PIPE_ALL);
    GlobalTensor<float,Shape<1,1,1,R,C>,pto::Stride<1,1,1,C,1>> native(
        reinterpret_cast<__gm__ float*>(stripe==0?first:second));
    TSTORE(native,a);pipe_barrier(PIPE_ALL);
    // Move residual weighting into this independent preparation step:
    // Gamma_i = exp(G_last-G_i) * K_i. The recurrence consumes Gamma^T*R.
    G decay,delta;TASSIGN(decay,164*1024);TASSIGN(delta,165*1024);
    float last=gates.GetValue(valid-1),last_low=lows.GetValue(valid-1);
    TMULS(decay,gates,-1.0f);PipeBarrierVec();TADDS(decay,decay,last);PipeBarrierVec();
    TMULS(delta,lows,-1.0f);PipeBarrierVec();TADDS(delta,delta,last_low);PipeBarrierVec();
    TADD(decay,decay,delta);PipeBarrierVec();TEXP(decay,decay);PipeBarrierVec();
    Tile<TileType::Vec,half,R,C> key;TASSIGN(key,128*1024);
    mega_vector_load<R,C>(key,original_k+((pos+stripe)*kh+key_head)*C,count,C,kh*C);
    TCVT(a,key,RoundMode::CAST_NONE);PipeBarrierVec();
    Col weight;TASSIGN(weight,164*1024+stripe*sizeof(float));
    TROWEXPANDMUL(a,a,weight);pipe_barrier(PIPE_ALL);
    if(count>0){
            auto storage=reinterpret_cast<__gm__ float*>(head<H/2?weighted_first:weighted_second);
            GlobalTensor<float,Shape<1,1,1,DYNAMIC,C>,pto::Stride<1,1,1,C,1>> gm(
                storage+((head%(H/2))*tokens+pos+stripe)*C,{1,1,1,count,C});
            Tile<TileType::Vec,float,R,C,BLayout::RowMajor,DYNAMIC,C> rows(count);TASSIGN(rows,96*1024);
            TSTORE(gm,rows);pipe_barrier(PIPE_ALL);
    }
    // Inverse joins use only the first 16 KiB of their 32 KiB workspace.
    // The second half holds two producer/consumer coupling flags.
    Tile<TileType::Vec,half,1,16> marker;TASSIGN(marker,168*1024);
    TEXPANDS(marker,static_cast<half>(0));pipe_barrier(PIPE_ALL);
    marker.SetValue(15,static_cast<half>(strongly_coupled?1:0));pipe_barrier(PIPE_ALL);
    GlobalTensor<half,Shape<1,1,1,1,16>,pto::Stride<1,1,1,16,1>> flag(control+get_subblockid()*16);
    TSTORE(flag,marker);pipe_barrier(PIPE_ALL);
#endif
}

template<int H,int C>
AICORE inline void mega_prepare_inverse(__gm__ float* g,__gm__ float* glow,__gm__ half* beta,
    __gm__ float* kk,__gm__ float* kk_upper,__gm__ float* qk,__gm__ float* lower,__gm__ float* full,
    __gm__ half* weighted_first,__gm__ half* attention_high,__gm__ half* attention_low,
    int64_t pos,int valid,int head,int kh,int64_t tokens,__gm__ half* first,__gm__ half* second,
    __gm__ half* original_k,__gm__ half* weighted_second,__gm__ half* control)
{
    bool coupled=mega_prepare_weights<H,C>(g,glow,beta,kk,kk_upper,qk,lower,full,attention_high,attention_low,pos,valid,head,kh,tokens);
    mega_finish_inverse<H,C>(g,glow,lower,full,weighted_first,pos,valid,head,kh,tokens,first,second,original_k,weighted_second,control,
        coupled);
}

// State is resident at [64,96) KiB in each vector subblock for the entire head.
template<int D>
AICORE inline void mega_initialize_state(__gm__ half* initial,__gm__ half* final_state,
                                         __gm__ float* exchange,bool has_initial,bool empty)
{
#if defined(__DAV_VEC__)
    constexpr int R=D/2;
    Tile<TileType::Vec,half,R,D> source;Tile<TileType::Vec,float,R,D> state;
    TASSIGN(source,0);TASSIGN(state,64*1024);
    if(has_initial)mega_vector_load<R,D>(source,initial+get_subblockid()*R*D,R,D,D);
    else{TEXPANDS(source,static_cast<half>(0));PipeBarrierVec();}
    if(empty){
        GlobalTensor<half,Shape<1,1,1,R,D>,pto::Stride<1,1,1,D,1>> gm(final_state+get_subblockid()*R*D);
        TSTORE(gm,source);pipe_barrier(PIPE_ALL);
    }else{
        TCVT(state,source,RoundMode::CAST_NONE);pipe_barrier(PIPE_ALL);
        auto pair=reinterpret_cast<__gm__ half*>(exchange)+get_subblockid()*R*D*2;
        mega_store_pair<R,D>(state,pair,pair+D,R,2*D);
    }
#endif
}

// Form B = beta * (V - exp(G) * K*S) before applying the inverse.
// Base output exp(G)*Q*S remains in UB while the cube solves for residuals.
struct MegaDecay { float alpha, log_alpha; };
template<int H,int C,int D>
AICORE inline MegaDecay mega_prepare_rhs(__gm__ half* v,__gm__ float* g,__gm__ float* glow,__gm__ half* beta,
    __gm__ float* projection,__gm__ float* output_base,
    int64_t pos,int valid,int head,int64_t tokens)
{
#if defined(__DAV_VEC__)
    constexpr int R=C/2,Gate=160*1024;
    Tile<TileType::Vec,float,1,C> gates,lows,exponentials,betas,correction;
    Tile<TileType::Vec,half,1,C> beta_h;
    TASSIGN(gates,Gate);TASSIGN(lows,Gate+512);TASSIGN(exponentials,Gate+1024);
    TASSIGN(betas,Gate+1536);TASSIGN(beta_h,Gate+2048);TASSIGN(correction,Gate+2560);
    mega_vector_load<1,C>(gates,g+head*tokens+pos,1,valid,C);
    mega_vector_load<1,C>(lows,glow+head*tokens+pos,1,valid,C);
    mega_vector_load<1,C>(beta_h,beta+head*tokens+pos,1,valid,C);
    TEXP(exponentials,gates);TCVT(betas,beta_h,RoundMode::CAST_NONE);PipeBarrierVec();
    TEXP(correction,lows);PipeBarrierVec();TMUL(exponentials,exponentials,correction);pipe_barrier(PIPE_ALL);
    float alpha=exponentials.GetValue(valid-1);
    float log_alpha=gates.GetValue(valid-1)+lows.GetValue(valid-1);
    int stripe=get_subblockid()*R,count=max(0,min(R,valid-stripe));
    Tile<TileType::Vec,float,R,D> value,values,base;
    Tile<TileType::Vec,half,R,D> input;
    Tile<TileType::Vec,float,R,1,BLayout::ColMajor,R,1> expcol,bcol;
    // [96,128) KiB holds the pending next-chunk coefficients. Convert V
    // before loading projections so both can reuse the first UB region.
    TASSIGN(value,0);TASSIGN(base,32*1024);TASSIGN(values,128*1024);TASSIGN(input,0);
    TASSIGN(expcol,Gate+1024+stripe*sizeof(float));TASSIGN(bcol,Gate+1536+stripe*sizeof(float));
    mega_vector_load<R,D>(input,v+((pos+stripe)*H+head)*D,count,D,H*D);
    TCVT(values,input,RoundMode::CAST_NONE);pipe_barrier(PIPE_ALL);
    mega_vector_load<R,D>(value,projection+stripe*D,R,D,D);
    TROWEXPANDMUL(value,value,expcol);pipe_barrier(PIPE_ALL);
    TSUB(value,values,value);PipeBarrierVec();
    TROWEXPANDMUL(value,value,bcol);pipe_barrier(PIPE_ALL);
    GlobalTensor<float,Shape<1,1,1,R,D>,pto::Stride<1,1,1,D,1>> saved(projection+C*D+stripe*D);
    TSTORE(saved,value);pipe_barrier(PIPE_ALL);
    mega_vector_load<R,D>(base,output_base+stripe*D,R,D,D);
    TROWEXPANDMUL(base,base,expcol);pipe_barrier(PIPE_ALL);
    return {alpha,log_alpha};
#else
    return {0.0f,0.0f};
#endif
}

template<int H,int C,int D>
AICORE inline void mega_finish_output(__gm__ float* local,__gm__ half* output,int valid)
{
#if defined(__DAV_VEC__)
    constexpr int R=C/2;
    int stripe=get_subblockid()*R,count=max(0,min(R,valid-stripe));
    Tile<TileType::Vec,float,R,D> value,base;Tile<TileType::Vec,half,R,D> result;
    TASSIGN(value,0);TASSIGN(base,32*1024);TASSIGN(result,96*1024);
    mega_vector_load<R,D>(value,local+stripe*D,R,D,D);
    TADD(value,value,base);PipeBarrierVec();TCVT(result,value,RoundMode::CAST_RINT);pipe_barrier(PIPE_ALL);
    if(count>0){
        Tile<TileType::Vec,half,R,D,BLayout::RowMajor,DYNAMIC,D> tail(count);TASSIGN(tail,96*1024);
        GlobalTensor<half,Shape<1,1,1,DYNAMIC,D>,pto::Stride<1,1,1,H*D,1>> gm(output+stripe*H*D,{1,1,1,count,D});
        TSTORE(gm,tail);pipe_barrier(PIPE_ALL);
    }
#endif
}

template<int D>
AICORE inline void mega_update_state(__gm__ float* exchange,MegaDecay decay,__gm__ float* saved_compensation,
                                     __gm__ half* final_state,bool first,bool last)
{
#if defined(__DAV_VEC__)
    constexpr int R=D/2;
    Tile<TileType::Vec,float,R,D> state,update;Tile<TileType::Vec,half,R,D> result;
    TASSIGN(state,64*1024);TASSIGN(update,0);TASSIGN(result,32*1024);
    int offset=get_subblockid()*R*D;
    GlobalTensor<float,Shape<1,1,1,R,D>,pto::Stride<1,1,1,D,1>> gm(exchange+offset);
    TLOAD(update,gm);pipe_barrier(PIPE_ALL);
    // Output's base tile is dead here. Preserve state rounding errors across
    // chunks in the two existing, otherwise unused WY scratch allocations.
    Tile<TileType::Vec,float,R,D> compensation,y,next;
    TASSIGN(compensation,32*1024);TASSIGN(y,96*1024);TASSIGN(next,128*1024);
    GlobalTensor<float,Shape<1,1,1,R,D>,pto::Stride<1,1,1,D,1>> cg(saved_compensation);
    if(first){TEXPANDS(compensation,0.0f);PipeBarrierVec();}
    else{TLOAD(compensation,cg);pipe_barrier(PIPE_ALL);}
    TMULS(compensation,compensation,decay.alpha);PipeBarrierVec();
    if(decay.log_alpha>=-0.125f && decay.log_alpha<=0.125f){
        TMULS(y,state,mega_small_expm1(decay.log_alpha));PipeBarrierVec();
        TSUB(y,y,compensation);PipeBarrierVec();TADD(next,state,y);PipeBarrierVec();
        TSUB(compensation,next,state);PipeBarrierVec();TSUB(compensation,compensation,y);PipeBarrierVec();
        TMOV(state,next);PipeBarrierVec();
    }else{TMULS(state,state,decay.alpha);PipeBarrierVec();}
    TSUB(y,update,compensation);PipeBarrierVec();TADD(next,state,y);PipeBarrierVec();
    TSUB(compensation,next,state);PipeBarrierVec();TSUB(compensation,compensation,y);PipeBarrierVec();
    TMOV(state,next);pipe_barrier(PIPE_ALL);
    TSTORE(cg,compensation);pipe_barrier(PIPE_ALL);
    // Interleaved high/low rows occupy exactly the FP32 update's footprint.
    // Each subblock overwrites only rows it has already loaded.
    auto pair=reinterpret_cast<__gm__ half*>(exchange)+2*offset;
    mega_store_pair<R,D>(state,pair,pair+D,R,2*D);
    if(last){
        TSUB(state,state,compensation);PipeBarrierVec();
        TCVT(result,state,RoundMode::CAST_RINT);pipe_barrier(PIPE_ALL);
        GlobalTensor<half,Shape<1,1,1,R,D>,pto::Stride<1,1,1,D,1>> fg(final_state+offset);
        TSTORE(fg,result);pipe_barrier(PIPE_ALL);
    }
#endif
}

// Reduce high/low products with error-free TwoSum at every tree node.
// The low tree also carries the persistent state compensation.
template<int Rows,int Cols>
AICORE inline void mega_sum_pairs()
{
#if defined(__DAV_VEC__)
    constexpr int HalfBytes=Rows/2*Cols*sizeof(float);
    using M=Tile<TileType::Vec,float,Rows/2,Cols>;
    M a,b,lo,lb,sum,z;
    TASSIGN(a,64*1024);TASSIGN(b,64*1024+HalfBytes);
    TASSIGN(lo,96*1024);TASSIGN(lb,96*1024+HalfBytes);
    TASSIGN(sum,128*1024);TASSIGN(z,128*1024+HalfBytes);
    TADD(sum,a,b);TADD(lb,lb,lo);PipeBarrierVec();
    TSUB(z,sum,a);PipeBarrierVec();TSUB(lo,sum,z);PipeBarrierVec();
    TSUB(lo,a,lo);TSUB(b,b,z);PipeBarrierVec();TADD(lo,lo,b);PipeBarrierVec();
    TADD(lo,lo,lb);TMOV(a,sum);PipeBarrierVec();
    if constexpr(Rows>2)mega_sum_pairs<Rows/2,Cols>();
#endif
}

template<int D,typename State,typename Key,typename Row>
AICORE inline void mega_accurate_dot(State& state,State& compensation,Key& key,Row& result,Row& error)
{
#if defined(__DAV_VEC__)
    constexpr int V=D/2;
    Tile<TileType::Vec,float,D,V> high,low,temp;
    TASSIGN(high,64*1024);TASSIGN(low,96*1024);TASSIGN(temp,128*1024);
    Tile<TileType::Vec,uint32_t,D,V> bits,split;
    TASSIGN(bits,(uint64_t)state.data());TASSIGN(split,64*1024);
    // Two 12-bit FP32 components retain the full exponent range. Their
    // products with the original 11-bit FP16 key/query are exact in FP32.
    TANDS(split,bits,0xfffff000u);PipeBarrierVec();
    TSUB(low,state,high);PipeBarrierVec();
    TROWEXPANDMUL(high,high,key);TROWEXPANDMUL(low,low,key);
    TROWEXPANDMUL(temp,compensation,key);PipeBarrierVec();
    TSUB(low,low,temp);PipeBarrierVec();
    mega_sum_pairs<D,V>();
    Tile<TileType::Vec,float,1,V> hi,lo;TASSIGN(hi,64*1024);TASSIGN(lo,96*1024);
    TMOV(result,hi);TMOV(error,lo);PipeBarrierVec();
#endif
}

// High-gain, strongly correlated updates amplify errors in a chunk solve.
// Preserve their state with compensated FP32 additions and balanced dot sums.
// Each vector subblock owns half the value columns and needs no token-wise sync.
template<int H,int D>
AICORE inline void mega_compensated_recurrence(__gm__ half* q,__gm__ half* k,__gm__ half* v,
    __gm__ float* g_raw,__gm__ half* beta,__gm__ half* initial,bool has_initial,__gm__ half* output,
    __gm__ half* final_state,int64_t start,int64_t end,int head,int kh,int64_t tokens)
{
#if defined(__DAV_VEC__)
    constexpr int V=D/2;
    int column=get_subblockid()*V;
    Tile<TileType::Vec,float,D,V> state,compensation,y,next;
    TASSIGN(state,0);TASSIGN(compensation,32*1024);TASSIGN(y,96*1024);TASSIGN(next,128*1024);
    if(has_initial){
        Tile<TileType::Vec,half,D,V> source;TASSIGN(source,96*1024);
        mega_vector_load<D,V>(source,initial+column,D,V,D);
        TCVT(state,source,RoundMode::CAST_NONE);PipeBarrierVec();
    }else{TEXPANDS(state,0.0f);PipeBarrierVec();}
    TEXPANDS(compensation,0.0f);PipeBarrierVec();
    Tile<TileType::Vec,float,1,D> keys,queries;
    Tile<TileType::Vec,float,D,1,BLayout::ColMajor,D,1> keycol,querycol;
    TASSIGN(keys,160*1024);TASSIGN(keycol,160*1024);
    TASSIGN(queries,160*1024+512);TASSIGN(querycol,160*1024+512);
    Tile<TileType::Vec,half,1,D> key_h,query_h;
    Tile<TileType::Vec,half,1,V> value_h,result_h;
    TASSIGN(key_h,161*1024);TASSIGN(query_h,161*1024+512);
    TASSIGN(value_h,162*1024);TASSIGN(result_h,162*1024);
    Tile<TileType::Vec,float,1,V> values,prediction,correction,delta;
    TASSIGN(values,162*1024+512);TASSIGN(prediction,163*1024);
    TASSIGN(correction,163*1024+512);TASSIGN(delta,164*1024);
    Tile<TileType::Vec,float,1,V> delta_low,virt,delta_high_part,delta_low_part;
    TASSIGN(delta_low,166*1024+512);TASSIGN(virt,167*1024);
    TASSIGN(delta_high_part,167*1024+512);TASSIGN(delta_low_part,168*1024);
    Tile<TileType::Vec,uint32_t,1,V> delta_bits,split_bits;
    TASSIGN(delta_bits,164*1024);TASSIGN(split_bits,167*1024+512);
    Tile<TileType::Vec,float,1,8> gate;
    Tile<TileType::Vec,half,1,16> beta_h;
    Tile<TileType::Vec,float,1,16> beta_f;
    TASSIGN(gate,165*1024);TASSIGN(beta_h,165*1024+512);TASSIGN(beta_f,166*1024);
    for(int64_t token=start;token<end;++token){
        int64_t input=(token*kh+head/(H/kh))*D;
        mega_vector_load<1,D>(key_h,k+input,1,D,D);
        mega_vector_load<1,D>(query_h,q+input,1,D,D);
        mega_vector_load<1,V>(value_h,v+(token*H+head)*D+column,1,V,V);
        mega_vector_load<1,8>(gate,g_raw+token*H+head,1,1,8);
        mega_vector_load<1,16>(beta_h,beta+head*tokens+token,1,1,16);
        TCVT(keys,key_h,RoundMode::CAST_NONE);TCVT(queries,query_h,RoundMode::CAST_NONE);
        TCVT(values,value_h,RoundMode::CAST_NONE);TCVT(beta_f,beta_h,RoundMode::CAST_NONE);
        pipe_barrier(PIPE_ALL);
        float log_decay=gate.GetValue(0);
        TEXP(gate,gate);pipe_barrier(PIPE_ALL);
        float decay=gate.GetValue(0),rate=beta_f.GetValue(0);
        TMULS(compensation,compensation,decay);PipeBarrierVec();
        if(log_decay!=0.0f && log_decay>=-0.125f && log_decay<=0.125f){
            // S*exp(g) = S + S*expm1(g). Preserve both the tiny decay
            // and its addition error, just as for the rank-one update.
            TMULS(y,state,mega_small_expm1(log_decay));PipeBarrierVec();
            TSUB(y,y,compensation);PipeBarrierVec();TADD(next,state,y);PipeBarrierVec();
            TSUB(compensation,next,state);PipeBarrierVec();TSUB(compensation,compensation,y);PipeBarrierVec();
            TMOV(state,next);PipeBarrierVec();
        }else{TMULS(state,state,decay);PipeBarrierVec();}
        mega_accurate_dot<D>(state,compensation,keycol,prediction,correction);
        // Error-free subtraction keeps the low projection bits through the
        // residual, rather than discarding them before a long update chain.
        TSUB(delta,values,prediction);PipeBarrierVec();
        TSUB(virt,delta,values);PipeBarrierVec();TSUB(delta_low,delta,virt);PipeBarrierVec();
        TSUB(delta_low,values,delta_low);TADD(virt,prediction,virt);PipeBarrierVec();
        TSUB(delta_low,delta_low,virt);PipeBarrierVec();TSUB(delta_low,delta_low,correction);PipeBarrierVec();
        // beta and K are original FP16 inputs. Splitting the FP32 operand
        // into 12-bit parts makes both products exact without FP16 narrowing.
        TANDS(split_bits,delta_bits,0xfffff000u);PipeBarrierVec();
        TSUB(delta_low_part,delta,delta_high_part);PipeBarrierVec();
        TMULS(delta_high_part,delta_high_part,rate);TMULS(delta_low_part,delta_low_part,rate);
        TMULS(delta_low,delta_low,rate);PipeBarrierVec();
        TADD(delta,delta_high_part,delta_low_part);PipeBarrierVec();
        TSUB(correction,delta_high_part,delta);PipeBarrierVec();TADD(correction,correction,delta_low_part);PipeBarrierVec();
        TADD(delta_low,delta_low,correction);PipeBarrierVec();
        TANDS(split_bits,delta_bits,0xfffff000u);PipeBarrierVec();
        TSUB(delta_low_part,delta,delta_high_part);PipeBarrierVec();
        Tile<TileType::Vec,float,D,V> update;TASSIGN(update,64*1024);
        TCOLEXPAND(update,delta_high_part);TCOLEXPAND(y,delta_low_part);TCOLEXPAND(next,delta_low);PipeBarrierVec();
        TROWEXPANDMUL(update,update,keycol);TROWEXPANDMUL(y,y,keycol);TROWEXPANDMUL(next,next,keycol);PipeBarrierVec();
        TADD(y,y,next);PipeBarrierVec();TSUB(compensation,compensation,y);PipeBarrierVec();
        // TwoSum(state, update) retains the addition error even when the
        // two large terms nearly cancel; compensation has the negative sign.
        TADD(next,state,update);PipeBarrierVec();TSUB(y,next,state);PipeBarrierVec();
        TSUB(update,update,y);PipeBarrierVec();TSUB(y,next,y);PipeBarrierVec();
        TSUB(y,state,y);PipeBarrierVec();TADD(y,y,update);PipeBarrierVec();
        TSUB(compensation,compensation,y);PipeBarrierVec();
        TMOV(state,next);PipeBarrierVec();
        mega_accurate_dot<D>(state,compensation,querycol,prediction,correction);
        TADD(prediction,prediction,correction);PipeBarrierVec();TCVT(result_h,prediction,RoundMode::CAST_RINT);pipe_barrier(PIPE_ALL);
        GlobalTensor<half,Shape<1,1,1,1,V>,pto::Stride<1,1,1,V,1>> out(output+(token*H+head)*D+column);
        TSTORE(out,result_h);pipe_barrier(PIPE_ALL);
    }
    TSUB(state,state,compensation);PipeBarrierVec();
    Tile<TileType::Vec,half,D,V> result;TASSIGN(result,96*1024);
    TCVT(result,state,RoundMode::CAST_RINT);pipe_barrier(PIPE_ALL);
    GlobalTensor<half,Shape<1,1,1,D,V>,pto::Stride<1,1,1,D,1>> fg(final_state+column);
    TSTORE(fg,result);pipe_barrier(PIPE_ALL);
    ffts_cross_core_sync(PIPE_MTE3,1|(2<<4)|(3<<8));
#endif
#if defined(__DAV_CUBE__)
    wait_flag_dev(3);ffts_cross_core_sync(PIPE_FIX,1|(2<<4)|(7<<8));
#endif
}

template<int H,int C,int D>
AICORE inline void mega_fused_pipeline(__gm__ half* q,__gm__ half* k,__gm__ half* v,
    __gm__ float* g,__gm__ float* glow,__gm__ half* beta,__gm__ half* weighted_first,__gm__ half* weighted_second,
    __gm__ half* att_high,__gm__ half* att_low,
    __gm__ half* uh_ws,__gm__ half* ul_ws,__gm__ half* k_ws,__gm__ half* state_ws,__gm__ half* solve_ws,
    __gm__ half* output,__gm__ half* final_state,__gm__ half* initial,bool has_initial,
    __gm__ int32_t* cu,int64_t sequences,int kh,int64_t tokens,
    __gm__ float* kk,__gm__ float* kk_upper,__gm__ float* qk,__gm__ float* lower,__gm__ float* full,__gm__ float* g_raw,
    __gm__ half* compensation_first,__gm__ half* compensation_second)
{
#if defined(__DAV_VEC__)
    set_mask_norm();set_vector_mask(-1,-1);
    Tile<TileType::Vec,uint32_t,1,16*(C/2)> indices;TASSIGN(indices,176*1024);
    // Byte offsets gather coefficients directly from the untransposed stripe.
    for(int j=0;j<16;++j)for(int col=0;col<C/2;++col)
        indices.SetValue(j*(C/2)+col,
            (((col/16)*16+j)*C+get_subblockid()*(C/2)+(col/16)*16)*sizeof(float));
    pipe_barrier(PIPE_ALL);
#endif
#if defined(__DAV_CUBE__)
    MegaPair<D,D> state;MegaFloat<C,C> attention,inverse;MegaFloat<C,D> native_rhs,native_residual,native_gamma;
    state.assign(0);TASSIGN(inverse,64*1024);TASSIGN(native_rhs,192*1024);TASSIGN(native_gamma,256*1024);
    TASSIGN(attention,384*1024);TASSIGN(native_residual,448*1024);
    typename MegaPair<2*C,D>::Mat inputs;TASSIGN(inputs,256*1024);
    TileAcc<float,2*C,D> projections;TASSIGN(projections,0);
    TileAcc<float,2*C,D,C,D> first_product,second_product;
    TASSIGN(first_product,0);TASSIGN(second_product,C*16*sizeof(float));
    TileAcc<float,C,D> acc;TASSIGN(acc,0);
    using Tail=Tile<TileType::Mat,half,2*C,D,BLayout::ColMajor,DYNAMIC,D,SLayout::RowMajor,512,PadValue::Zero>;
#endif
    int block=get_block_idx();auto exchange=reinterpret_cast<__gm__ float*>(state_ws+block*4*D*D);
    auto local_output=reinterpret_cast<__gm__ float*>(k_ws+block*2*C*D);
    auto first=uh_ws+block*C*C,second=ul_ws+block*C*C;
    auto solve_scratch=reinterpret_cast<__gm__ float*>(solve_ws+block*C*C);
    auto control=solve_ws+block*C*C+C*C/2;
#if defined(__DAV_VEC__)
    auto saved_compensation=reinterpret_cast<__gm__ float*>(
        (get_subblockid()==0?compensation_first:compensation_second)+block*D*D);
#endif
    for(int64_t work=block;work<sequences*H;work+=get_block_num()){
        int seq=work/H,head=work%H;int64_t start=cu[seq],end=cu[seq+1];
#if defined(__DAV_VEC__)
        if(start<end)mega_prepare_inverse<H,C>(g,glow,beta,kk,kk_upper,qk,lower,full,
            weighted_first,att_high,att_low,start,min(static_cast<int64_t>(C),end-start),head,kh,tokens,first,second,k,weighted_second,control);
        mega_initialize_state<D>(has_initial?initial+work*D*D:initial,final_state+work*D*D,exchange,has_initial,start==end);
#endif
        if(start==end)continue;
#if defined(__DAV_VEC__)
        ffts_cross_core_sync(PIPE_MTE3,1|(2<<4)|(1<<8));
#endif
#if defined(__DAV_CUBE__)
        wait_flag_dev(1);
        auto sh=reinterpret_cast<__gm__ half*>(exchange);
        mega_load_pair<D,D>(state,sh,sh+D,D,2*D);
        ffts_cross_core_sync(PIPE_MTE2,1|(2<<4)|(7<<8));
#endif
        for(int64_t pos=start;pos<end;pos+=C){
            int valid=min(static_cast<int64_t>(C),end-pos);
#if defined(__DAV_VEC__)
            wait_flag_dev(7);
#endif
            // DMA overwrites this reused slot; discard the previous scalar cache line.
            dcci(control,SINGLE_CACHE_LINE);
            auto markers=reinterpret_cast<__gm__ volatile uint16_t*>(control);
            bool strongly_coupled=(markers[15]&0x7fff)!=0;
            if(valid>C/2)strongly_coupled|=(markers[31]&0x7fff)!=0;
            if(strongly_coupled){
                // Restart from the supplied initial state: a less precise prefix
                // must not be amplified by a later strongly coupled update.
                mega_compensated_recurrence<H,D>(q,k,v,g_raw,beta,
                    has_initial?initial+work*D*D:initial,has_initial,output,final_state+work*D*D,
                    start,end,head,kh,tokens);
                break;
            }
#if defined(__DAV_VEC__)
            bool next_coupled=false;
            if(pos+C<end)next_coupled=mega_prepare_weights<H,C>(g,glow,beta,kk,kk_upper,qk,lower,full,att_high,att_low,pos+C,min(static_cast<int64_t>(C),end-pos-C),head,
                kh,tokens);
            wait_flag_dev(5);
            auto decay=mega_prepare_rhs<H,C,D>(v,g,glow,beta,exchange,local_output,pos,valid,head,tokens);
            ffts_cross_core_sync(PIPE_MTE3,1|(2<<4)|(6<<8));
            if(pos+C<end)mega_finish_inverse<H,C>(g,glow,lower,full,weighted_first,pos+C,min(static_cast<int64_t>(C),end-pos-C),head,kh,tokens,first,second,
                k,weighted_second,control,next_coupled);
            wait_flag_dev(9);
            mega_finish_output<H,C,D>(local_output,output+(pos*H+head)*D,valid);
            mega_update_state<D>(exchange,decay,saved_compensation,final_state+work*D*D,pos==start,pos+C>=end);
            ffts_cross_core_sync(PIPE_MTE3,1|(2<<4)|(6<<8));
#endif
#if defined(__DAV_CUBE__)
            Tail ktail(valid),qtail(valid);TASSIGN(ktail,256*1024);TASSIGN(qtail,256*1024+C*16*sizeof(half));
            using InputG=GlobalTensor<half,Shape<1,1,1,DYNAMIC,D>,pto::Stride<1,1,1,DYNAMIC,1>>;
            int64_t input_offset=(pos*kh+head/(H/kh))*D;
            InputG kgm(k+input_offset,{1,1,1,valid,D},{1,1,1,kh*D,1}),qgm(q+input_offset,{1,1,1,valid,D},{1,1,1,kh*D,1});
            TLOAD(ktail,kgm);pipe_barrier(PIPE_ALL);
            // Pad the whole stacked tile before Q is loaded into its lower
            // half. This clears both tails without touching valid Q rows.
            if(valid!=C){TFILLPAD(ktail,ktail);pipe_barrier(PIPE_ALL);}
            TLOAD(qtail,qgm);pipe_barrier(PIPE_ALL);
            mega_input_pair_product<2*C,D,D>(projections,inputs,state);
            GlobalTensor<float,Shape<1,1,1,C,D>,pto::Stride<1,1,1,D,1>> ex(exchange),og(local_output);
            TSTORE(ex,first_product);TSTORE(og,second_product);pipe_barrier(PIPE_ALL);
            Tile<TileType::Mat,float,C,C,BLayout::ColMajor,C/2,C,SLayout::RowMajor,512> first_half,second_half;
            TASSIGN(first_half,64*1024);TASSIGN(second_half,64*1024+(C/2)*8*sizeof(float));
            GlobalTensor<float,Shape<1,1,1,C/2,C>,pto::Stride<1,1,1,C,1>> fg(reinterpret_cast<__gm__ float*>(first)),sg(reinterpret_cast<__gm__ float*>(second));
            TLOAD(first_half,fg);TLOAD(second_half,sg);pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_FIX,1|(2<<4)|(5<<8));
            mega_float_inverse<C>(inverse,solve_scratch);
            {
                auto storage=reinterpret_cast<__gm__ float*>(head<H/2?att_high:att_low);
                GlobalTensor<float,Shape<1,1,1,DYNAMIC,C>,pto::Stride<1,1,1,C,1>> ag(
                    storage+((head%(H/2))*tokens+pos)*C,{1,1,1,valid,C});
                Tile<TileType::Mat,float,C,C,BLayout::ColMajor,DYNAMIC,C,SLayout::RowMajor,512,PadValue::Zero> part(valid);
                TASSIGN(part,384*1024);TLOAD(part,ag);pipe_barrier(PIPE_ALL);
                if(valid!=C){TFILLPAD(part,part);pipe_barrier(PIPE_ALL);}
            }
            wait_flag_dev(6);
            GlobalTensor<float,Shape<1,1,1,C,D>,pto::Stride<1,1,1,D,1>> bg(exchange+C*D);
            TLOAD(native_rhs,bg);pipe_barrier(PIPE_ALL);
            TileLeft<float,C,C> invleft;TileRight<float,C,D> valright;TASSIGN(invleft,0);TASSIGN(valright,0);
            TMOV(invleft,inverse);TMOV(valright,native_rhs);pipe_barrier(PIPE_ALL);
            TMATMUL(acc,invleft,valright);pipe_barrier(PIPE_ALL);
            TSTORE(ex,acc);pipe_barrier(PIPE_ALL);
            TLOAD(native_residual,ex);pipe_barrier(PIPE_ALL);
            {
                auto storage=reinterpret_cast<__gm__ float*>(head<H/2?weighted_first:weighted_second);
                GlobalTensor<float,Shape<1,1,1,DYNAMIC,C>,pto::Stride<1,1,1,C,1>> gg(
                    storage+((head%(H/2))*tokens+pos)*C,{1,1,1,valid,C});
                Tile<TileType::Mat,float,C,C,BLayout::ColMajor,DYNAMIC,C,SLayout::RowMajor,512,PadValue::Zero> part(valid);
                TASSIGN(part,256*1024);TLOAD(part,gg);pipe_barrier(PIPE_ALL);
                if(valid!=C){TFILLPAD(part,part);pipe_barrier(PIPE_ALL);}
            }
            using GT=Tile<TileType::Mat,float,D,C,BLayout::RowMajor,D,C,SLayout::ColMajor,512>;
            GT transpose;TRESHAPE(transpose,native_gamma);
            mega_combined_residual_products<C,D>(projections,attention,transpose,native_residual);
            TSTORE(og,first_product);TSTORE(ex,second_product);pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_FIX,1|(2<<4)|(9<<8));
            wait_flag_dev(6);mega_load_pair<D,D>(state,sh,sh+D,D,2*D);
            ffts_cross_core_sync(PIPE_MTE2,1|(2<<4)|(7<<8));
#endif
        }
#if defined(__DAV_VEC__)
        wait_flag_dev(7);
#endif
    }
    pipe_barrier(PIPE_ALL);
}
