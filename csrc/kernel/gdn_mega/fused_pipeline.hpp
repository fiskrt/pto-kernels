#pragma once

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
// The next chunk's inverse can be prepared while the cube solves this chunk.
template<int H,int C>
AICORE inline void mega_prepare_inverse(__gm__ float* g,__gm__ float* glow,__gm__ half* beta,
    __gm__ float* kk,__gm__ float* kk_upper,__gm__ float* qk,__gm__ float* lower,__gm__ float* full,
    __gm__ half* inverse_high,__gm__ half* inverse_low,__gm__ half* attention_high,__gm__ half* attention_low,
    int64_t pos,int valid,int head,int kh,int64_t tokens)
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
    M a,raw,mask;TASSIGN(a,0);TASSIGN(raw,96*1024);TASSIGN(mask,128*1024);
    Col gcol,lcol,bcol;
    TASSIGN(gcol,Gate+stripe*sizeof(float));TASSIGN(lcol,Gate+512+stripe*sizeof(float));
    TASSIGN(bcol,Gate+1024+stripe*sizeof(float));
    // Recompute decay coefficients for the second product rather than spill
    // either live state or output. All differences remain in FP32.
    for(int product=0;product<2;++product){
        TCOLEXPAND(raw,gates);PipeBarrierVec();TMULS(raw,raw,-1.0f);PipeBarrierVec();
        TROWEXPANDADD(raw,raw,gcol);PipeBarrierVec();
        TCOLEXPAND(a,lows);PipeBarrierVec();TMULS(a,a,-1.0f);PipeBarrierVec();
        TROWEXPANDADD(a,a,lcol);PipeBarrierVec();TADD(raw,raw,a);PipeBarrierVec();
        mega_vector_load<R,C>(a,full+stripe*C,R,C,C);
        TMUL(raw,raw,a);PipeBarrierVec();TEXP(raw,raw);PipeBarrierVec();TMUL(raw,raw,a);pipe_barrier(PIPE_ALL);
        if(product==0){
            mega_vector_load<R,C>(a,qk+((pos+stripe)*kh+key_head)*C,count,C,kh*C);
            TMUL(a,a,raw);pipe_barrier(PIPE_ALL);
            mega_store_pair<R,C>(a,attention_high+((pos+stripe)*H+head)*C,
                                   attention_low+((pos+stripe)*H+head)*C,count,H*C);
        }else{
            mega_load_kk<R,C>(a,kk+((pos+stripe)*kh+key_head)*(C/2),
                               kk_upper+((pos+stripe)*kh+key_head)*(C/2),count,kh*C/2);
            TMUL(a,a,raw);PipeBarrierVec();
        }
    }
    mega_vector_load<R,C>(mask,lower+stripe*C,R,C,C);
    TMUL(a,a,mask);PipeBarrierVec();TROWEXPANDMUL(a,a,bcol);PipeBarrierVec();TMULS(a,a,-1.0f);PipeBarrierVec();
    TABS(raw,a);PipeBarrierVec();
    Tile<TileType::Vec,float,1,R*C> flattened;TASSIGN(flattened,96*1024);
    Tile<TileType::Vec,float,1,8> maximum;
    Tile<TileType::Vec,float,1,1024> reduction;
    TASSIGN(maximum,170*1024);TASSIGN(reduction,171*1024);
    TROWMAX(maximum,flattened,reduction);pipe_barrier(PIPE_ALL);
    bool strongly_coupled=maximum.GetValue(0)>0.5f;
    mega_vector_load<R,C>(raw,full+stripe*C,R,C,C);TSUB(raw,raw,mask);PipeBarrierVec();
    mega_diagonal_inverses(a,raw,stripe);pipe_barrier(PIPE_ALL);
    // This strictly upper-triangular slot is otherwise exactly zero. A marked
    // chunk takes compensated recurrence and never uses this inverse matrix.
    if(strongly_coupled){a.SetValue(C-1,1.0f);pipe_barrier(PIPE_ALL);}
    mega_store_pair<R,C>(a,inverse_high+((pos+stripe)*H+head)*C,
                           inverse_low+((pos+stripe)*H+head)*C,count,H*C);
#endif
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
template<int H,int C,int D>
AICORE inline float mega_prepare_rhs(__gm__ half* v,__gm__ float* g,__gm__ float* glow,__gm__ half* beta,
    __gm__ float* projection,__gm__ float* output_base,__gm__ half* rh,__gm__ half* rl,
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
    int stripe=get_subblockid()*R,count=max(0,min(R,valid-stripe));
    Tile<TileType::Vec,float,R,D> value,values,base;
    Tile<TileType::Vec,half,R,D> input;
    Tile<TileType::Vec,float,R,1,BLayout::ColMajor,R,1> expcol,bcol;
    TASSIGN(value,0);TASSIGN(base,32*1024);TASSIGN(values,96*1024);TASSIGN(input,128*1024);
    TASSIGN(expcol,Gate+1024+stripe*sizeof(float));TASSIGN(bcol,Gate+1536+stripe*sizeof(float));
    mega_vector_load<R,D>(value,projection+stripe*D,R,D,D);
    TROWEXPANDMUL(value,value,expcol);pipe_barrier(PIPE_ALL);
    mega_vector_load<R,D>(input,v+((pos+stripe)*H+head)*D,count,D,H*D);
    TCVT(values,input,RoundMode::CAST_NONE);PipeBarrierVec();TSUB(value,values,value);PipeBarrierVec();
    TROWEXPANDMUL(value,value,bcol);pipe_barrier(PIPE_ALL);
    mega_store_pair<R,D>(value,rh+stripe*D,rl+stripe*D);
    mega_vector_load<R,D>(base,output_base+stripe*D,R,D,D);
    TROWEXPANDMUL(base,base,expcol);pipe_barrier(PIPE_ALL);
    Tile<TileType::Vec,float,1,R> grow,lrow,decay,delta;
    TASSIGN(grow,Gate+stripe*sizeof(float));TASSIGN(lrow,Gate+512+stripe*sizeof(float));
    TASSIGN(decay,164*1024);TASSIGN(delta,165*1024);
    float last=gates.GetValue(valid-1),last_low=lows.GetValue(valid-1);
    TMULS(decay,grow,-1.0f);PipeBarrierVec();TADDS(decay,decay,last);PipeBarrierVec();
    TMULS(delta,lrow,-1.0f);PipeBarrierVec();TADDS(delta,delta,last_low);PipeBarrierVec();
    TADD(decay,decay,delta);PipeBarrierVec();TEXP(decay,decay);pipe_barrier(PIPE_ALL);
    return alpha;
#else
    return 0.0f;
#endif
}

template<int C,int D>
AICORE inline void mega_scale_residual(__gm__ float* residual,__gm__ half* high,__gm__ half* low,int valid)
{
#if defined(__DAV_VEC__)
    constexpr int R=C/2;
    int stripe=get_subblockid()*R;
    Tile<TileType::Vec,float,R,1,BLayout::ColMajor,R,1> column;TASSIGN(column,164*1024);
    Tile<TileType::Vec,float,R,D> value;TASSIGN(value,0);
    mega_vector_load<R,D>(value,residual+stripe*D,R,D,D);
    TROWEXPANDMUL(value,value,column);pipe_barrier(PIPE_ALL);
    mega_store_pair<R,D>(value,high+stripe*D,low+stripe*D);
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
AICORE inline void mega_update_state(__gm__ float* exchange,float alpha,__gm__ half* final_state,bool last)
{
#if defined(__DAV_VEC__)
    constexpr int R=D/2;
    Tile<TileType::Vec,float,R,D> state,update;Tile<TileType::Vec,half,R,D> result;
    TASSIGN(state,64*1024);TASSIGN(update,0);TASSIGN(result,32*1024);
    int offset=get_subblockid()*R*D;
    GlobalTensor<float,Shape<1,1,1,R,D>,pto::Stride<1,1,1,D,1>> gm(exchange+offset);
    TLOAD(update,gm);pipe_barrier(PIPE_ALL);
    TMULS(state,state,alpha);PipeBarrierVec();TADD(state,state,update);pipe_barrier(PIPE_ALL);
    // Interleaved high/low rows occupy exactly the FP32 update's footprint.
    // Each subblock overwrites only rows it has already loaded.
    auto pair=reinterpret_cast<__gm__ half*>(exchange)+2*offset;
    mega_store_pair<R,D>(state,pair,pair+D,R,2*D);
    if(last){
        TCVT(result,state,RoundMode::CAST_RINT);pipe_barrier(PIPE_ALL);
        GlobalTensor<half,Shape<1,1,1,R,D>,pto::Stride<1,1,1,D,1>> fg(final_state+offset);
        TSTORE(fg,result);pipe_barrier(PIPE_ALL);
    }
#endif
}

// Pairwise reduction matches a fixed balanced tree, without a serial 128-row
// accumulation chain. The source may be destroyed after its elementwise product.
template<int Rows,int Cols>
AICORE inline void mega_sum_rows(uint32_t address)
{
#if defined(__DAV_VEC__)
    Tile<TileType::Vec,float,Rows/2,Cols> top,bottom;
    TASSIGN(top,address);TASSIGN(bottom,address+(Rows/2)*Cols*sizeof(float));
    TADD(top,top,bottom);PipeBarrierVec();
    if constexpr(Rows>2)mega_sum_rows<Rows/2,Cols>(address);
#endif
}

template<int D,typename State,typename Key,typename Row>
AICORE inline void mega_vector_dot(State& state,Key& key,Row& result)
{
#if defined(__DAV_VEC__)
    constexpr int V=D/2;
    Tile<TileType::Vec,float,D,V> product;TASSIGN(product,64*1024);
    Tile<TileType::Vec,float,1,V> sum;TASSIGN(sum,64*1024);
    TROWEXPANDMUL(product,state,key);PipeBarrierVec();
    mega_sum_rows<D,V>(64*1024);TMOV(result,sum);PipeBarrierVec();
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
        TEXP(gate,gate);pipe_barrier(PIPE_ALL);
        float decay=gate.GetValue(0),rate=beta_f.GetValue(0);
        TMULS(state,state,decay);TMULS(compensation,compensation,decay);PipeBarrierVec();
        mega_vector_dot<D>(state,keycol,prediction);mega_vector_dot<D>(compensation,keycol,correction);
        TSUB(prediction,prediction,correction);PipeBarrierVec();TSUB(delta,values,prediction);PipeBarrierVec();
        TMULS(delta,delta,rate);PipeBarrierVec();
        Tile<TileType::Vec,float,D,V> update;TASSIGN(update,64*1024);
        TCOLEXPAND(update,delta);PipeBarrierVec();TROWEXPANDMUL(update,update,keycol);PipeBarrierVec();
        TSUB(y,update,compensation);PipeBarrierVec();TADD(next,state,y);PipeBarrierVec();
        TSUB(compensation,next,state);PipeBarrierVec();TSUB(compensation,compensation,y);PipeBarrierVec();
        TMOV(state,next);PipeBarrierVec();
        mega_vector_dot<D>(state,querycol,prediction);mega_vector_dot<D>(compensation,querycol,correction);
        TSUB(prediction,prediction,correction);PipeBarrierVec();TCVT(result_h,prediction,RoundMode::CAST_RINT);pipe_barrier(PIPE_ALL);
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
    __gm__ float* g,__gm__ float* glow,__gm__ half* beta,__gm__ half* inverse_high,__gm__ half* inverse_low,
    __gm__ half* att_high,__gm__ half* att_low,__gm__ half* neg_i,
    __gm__ half* wh_ws,__gm__ half* wl_ws,__gm__ half* uh_ws,__gm__ half* ul_ws,__gm__ half* k_ws,__gm__ half* state_ws,
    __gm__ half* output,__gm__ half* final_state,__gm__ half* initial,bool has_initial,
    __gm__ int32_t* cu,int64_t sequences,int kh,int64_t tokens,
    __gm__ float* kk,__gm__ float* kk_upper,__gm__ float* qk,__gm__ float* lower,__gm__ float* full,__gm__ float* g_raw)
{
#if defined(__DAV_VEC__)
    set_mask_norm();set_vector_mask(-1,-1);
    Tile<TileType::Vec,uint32_t,1,16*(C/2)> indices;TASSIGN(indices,176*1024);
    for(int j=0;j<16;++j)for(int col=0;col<C/2;++col)indices.SetValue(j*(C/2)+col,(col/16)*16+j);
    pipe_barrier(PIPE_ALL);
#endif
#if defined(__DAV_CUBE__)
    MegaPair<D,D> state;MegaPair<C,C> inverse,attention;MegaPair<C,D> rhs,scaled,residual;
    state.assign(0);inverse.assign(64*1024);scaled.assign(128*1024);rhs.assign(192*1024);
    attention.assign(384*1024);residual.assign(448*1024);
    typename MegaPair<C,D>::Mat key,query;
    typename MegaPair<D,D>::Mat negative_identity;
    TASSIGN(key,256*1024);TASSIGN(query,320*1024);TASSIGN(negative_identity,480*1024);
    TileAcc<float,C,D> acc,other;TASSIGN(acc,0);TASSIGN(other,64*1024);
    using Tail=Tile<TileType::Mat,half,C,D,BLayout::ColMajor,DYNAMIC,D,SLayout::RowMajor,512,PadValue::Zero>;
#endif
    int block=get_block_idx();auto exchange=reinterpret_cast<__gm__ float*>(state_ws+block*4*D*D);
    auto local_output=reinterpret_cast<__gm__ float*>(k_ws+block*2*C*D);
    auto rh=wh_ws+block*C*D,rl=wl_ws+block*C*D;
    for(int64_t work=block;work<sequences*H;work+=get_block_num()){
        int seq=work/H,head=work%H;int64_t start=cu[seq],end=cu[seq+1];
#if defined(__DAV_VEC__)
        if(start<end)mega_prepare_inverse<H,C>(g,glow,beta,kk,kk_upper,qk,lower,full,
            inverse_high,inverse_low,att_high,att_low,start,min(static_cast<int64_t>(C),end-start),head,kh,tokens);
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
            auto marker=(pos*H+head)*C+C-1;
            auto markers=reinterpret_cast<__gm__ uint16_t*>(inverse_high);
            bool strongly_coupled=(markers[marker]&0x7fff)!=0;
            if(valid>C/2)strongly_coupled|=(markers[marker+(C/2)*H*C]&0x7fff)!=0;
            if(strongly_coupled){
                // Restart from the supplied initial state: a less precise prefix
                // must not be amplified by a later strongly coupled update.
                mega_compensated_recurrence<H,D>(q,k,v,g_raw,beta,
                    has_initial?initial+work*D*D:initial,has_initial,output,final_state+work*D*D,
                    start,end,head,kh,tokens);
                break;
            }
#if defined(__DAV_VEC__)
            wait_flag_dev(5);
            float alpha=mega_prepare_rhs<H,C,D>(v,g,glow,beta,exchange,local_output,rh,rl,pos,valid,head,tokens);
            ffts_cross_core_sync(PIPE_MTE3,1|(2<<4)|(6<<8));
            if(pos+C<end)mega_prepare_inverse<H,C>(g,glow,beta,kk,kk_upper,qk,lower,full,
                inverse_high,inverse_low,att_high,att_low,pos+C,min(static_cast<int64_t>(C),end-pos-C),head,kh,tokens);
            wait_flag_dev(3);
            mega_scale_residual<C,D>(exchange,rh,rl,valid);
            ffts_cross_core_sync(PIPE_MTE3,1|(2<<4)|(4<<8));
            wait_flag_dev(9);
            mega_finish_output<H,C,D>(local_output,output+(pos*H+head)*D,valid);
            wait_flag_dev(10);
            mega_update_state<D>(exchange,alpha,final_state+work*D*D,pos+C>=end);
            ffts_cross_core_sync(PIPE_MTE3,1|(2<<4)|(6<<8));
#endif
#if defined(__DAV_CUBE__)
            Tail ktail(valid),qtail(valid);TASSIGN(ktail,256*1024);TASSIGN(qtail,320*1024);
            using InputG=GlobalTensor<half,Shape<1,1,1,DYNAMIC,D>,pto::Stride<1,1,1,DYNAMIC,1>>;
            int64_t input_offset=(pos*kh+head/(H/kh))*D;
            InputG kgm(k+input_offset,{1,1,1,valid,D},{1,1,1,kh*D,1}),qgm(q+input_offset,{1,1,1,valid,D},{1,1,1,kh*D,1});
            TLOAD(ktail,kgm);TLOAD(qtail,qgm);pipe_barrier(PIPE_ALL);
            if(valid!=C){TFILLPAD(ktail,ktail);TFILLPAD(qtail,qtail);pipe_barrier(PIPE_ALL);}
            mega_input_pair_product<C,D,D>(acc,key,state);
            mega_input_pair_product<C,D,D>(other,query,state);
            GlobalTensor<float,Shape<1,1,1,C,D>,pto::Stride<1,1,1,D,1>> ex(exchange),og(local_output);
            TSTORE(ex,acc);TSTORE(og,other);pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_FIX,1|(2<<4)|(5<<8));
            auto offset=(pos*H+head)*C;
            mega_load_pair<C,C>(inverse,inverse_high+offset,inverse_low+offset,valid,H*C);
            mega_complete_inverse<C>(inverse,neg_i);
            mega_load_pair<C,C>(attention,att_high+offset,att_low+offset,valid,H*C);
            mega_load_packed<D,D,D>(negative_identity,negative_identity,neg_i,D,D);pipe_barrier(PIPE_ALL);
            wait_flag_dev(6);mega_load_pair<C,D>(rhs,rh,rl);
            mega_pair_product<C,C,D>(acc,inverse,rhs);
            TSTORE(ex,acc);pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_FIX,1|(2<<4)|(3<<8));
            mega_pair_from_acc<C,D>(residual,acc,negative_identity);
            mega_pair_product<C,C,D>(other,attention,residual);
            TSTORE(og,other);pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_FIX,1|(2<<4)|(9<<8));
            wait_flag_dev(4);mega_load_pair<C,D>(scaled,rh,rl);
            using Trans=Tile<TileType::Mat,half,D,C,BLayout::RowMajor,D,C,SLayout::ColMajor,512>;
            Trans transpose;TRESHAPE(transpose,key);
            mega_input_pair_product<D,C,D>(acc,transpose,scaled);
            TSTORE(ex,acc);pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_FIX,1|(2<<4)|(10<<8));
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
