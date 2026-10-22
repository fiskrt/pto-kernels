// Scale the RHS before multiplying by the inverse. Vector preparation depends
// only on K/V/beta/g, so it can run concurrently with the cube inverse stage.
template <int H, int C, int D>
AICORE inline void mega_prepare_wy(__gm__ half* k, __gm__ half* v, __gm__ float* g,
                                  __gm__ half* beta, __gm__ half* w, __gm__ half* u,
                                  __gm__ int32_t* cu, uint32_t matrices, int kh, int64_t tokens, uint32_t first = UINT32_MAX, uint32_t last = UINT32_MAX)
{
#if defined(__DAV_VEC__)
    set_mask_norm(); set_vector_mask(-1, -1);
    constexpr int R = C / 2, F = R * D * sizeof(float);
    using Shape2 = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Stride2 = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using Half = Tile<TileType::Vec, half, R, D, BLayout::RowMajor, R, D>;
    using HalfPart = Tile<TileType::Vec, half, R, D, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using Float = Tile<TileType::Vec, float, R, D, BLayout::RowMajor, R, D>;
    using Row = Tile<TileType::Vec, float, 1, R, BLayout::RowMajor, 1, R>;
    using RowPart = Tile<TileType::Vec, float, 1, R, BLayout::RowMajor, 1, DYNAMIC>;
    using Beta = Tile<TileType::Vec, half, 1, R, BLayout::RowMajor, 1, R>;
    using BetaPart = Tile<TileType::Vec, half, 1, R, BLayout::RowMajor, 1, DYNAMIC>;
    using Column = Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, R, 1>;
    Half key_h, value_h;
    Float key_f, value_f, coeff;
    Row gates, betas;
    Beta beta_h;
    TASSIGN(key_h, 0); TASSIGN(value_h, F / 2);
    TASSIGN(key_f, F); TASSIGN(value_f, 2 * F); TASSIGN(coeff, 3 * F);
    TASSIGN(gates, 4 * F); TASSIGN(betas, 4 * F + R * sizeof(float));
    TASSIGN(beta_h, 4 * F + 2 * R * sizeof(float));
    for (uint32_t m = first == UINT32_MAX ? get_block_idx() : first;
         m < (last == UINT32_MAX ? matrices : last);
         m += first == UINT32_MAX ? get_block_num() : matrices) {
        auto info = mk_solve::GetBSNDVarlenTileInfoFromCuSeqlens(m, H, C, cu);
        int stripe = get_subblockid() * R;
        int count = min(R, static_cast<int>(info.valid_size) - stripe);
        if (count <= 0) continue;
        int head = m % H;
        int64_t pos = (info.bsnd_offset - head * C) / (H * C) + stripe;
        HalfPart key_part(count, D), value_part(count, D);
        RowPart gpart(count); BetaPart bpart(count);
        TASSIGN(key_part, 0); TASSIGN(value_part, F / 2);
        TASSIGN(gpart, 4 * F); TASSIGN(bpart, 4 * F + 2 * R * sizeof(float));
        GlobalTensor<half, Shape2, Stride2> kgm(k + (pos * kh + head / (H / kh)) * D,
            {1, 1, 1, count, D}, {1, 1, 1, kh * D, 1});
        GlobalTensor<half, Shape2, Stride2> vgm(v + (pos * H + head) * D,
            {1, 1, 1, count, D}, {1, 1, 1, H * D, 1});
        GlobalTensor<float, Shape2, Stride2> ggm(g + head * tokens + pos,
            {1, 1, 1, 1, count}, {1, 1, 1, count, 1});
        GlobalTensor<half, Shape2, Stride2> bgm(beta + head * tokens + pos,
            {1, 1, 1, 1, count}, {1, 1, 1, count, 1});
        TLOAD(key_part, kgm); TLOAD(value_part, vgm); TLOAD(gpart, ggm); TLOAD(bpart, bgm);
        pipe_barrier(PIPE_ALL);
        TCVT(key_f, key_h, RoundMode::CAST_NONE);
        TCVT(value_f, value_h, RoundMode::CAST_NONE);
        TCVT(betas, beta_h, RoundMode::CAST_NONE); PipeBarrierVec();
        Column col; TASSIGN(col, 4 * F + R * sizeof(float));
        TROWEXPAND(coeff, col); PipeBarrierVec();
        TMUL(value_f, value_f, coeff); PipeBarrierVec();
        TCVT(value_h, value_f, RoundMode::CAST_NONE);
        TEXP(gates, gates); PipeBarrierVec();
        TMUL(betas, betas, gates); PipeBarrierVec();
        TROWEXPAND(coeff, col); PipeBarrierVec();
        TMUL(key_f, key_f, coeff); PipeBarrierVec();
#ifdef GDN_CUBE_STATE
        TMULS(key_f, key_f, -1.0f); PipeBarrierVec();
#endif
        TCVT(key_h, key_f, RoundMode::CAST_NONE);
        pipe_barrier(PIPE_ALL);
        GlobalTensor<half, Shape2, Stride2> wgm(w + (pos * H + head) * D,
            {1, 1, 1, count, D}, {1, 1, 1, H * D, 1});
        GlobalTensor<half, Shape2, Stride2> ugm(u + (pos * H + head) * D,
            {1, 1, 1, count, D}, {1, 1, 1, H * D, 1});
        TSTORE(wgm, key_part); TSTORE(ugm, value_part); pipe_barrier(PIPE_ALL);
    }
#endif
}

template <int H, int C, int D>
AICORE inline void mega_wy_matmul(__gm__ half* inverse, __gm__ half* w, __gm__ half* u,
                                 __gm__ int32_t* cu, uint32_t matrices)
{
#if defined(__DAV_CUBE__)
    using Mat = Tile<TileType::Mat, half, C, D, BLayout::ColMajor, C, D, SLayout::RowMajor, 512>;
    using Tail = Tile<TileType::Mat, half, C, D, BLayout::ColMajor, DYNAMIC, DYNAMIC,
                      SLayout::RowMajor, 512, PadValue::Zero>;
    using Inv = Tile<TileType::Mat, half, C, C, BLayout::ColMajor, C, C, SLayout::RowMajor, 512>;
    using InvTail = Tile<TileType::Mat, half, C, C, BLayout::ColMajor, DYNAMIC, DYNAMIC,
                         SLayout::RowMajor, 512, PadValue::Zero>;
    using Shape2 = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Stride2 = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    Inv inv; Mat wm, um;
    TASSIGN(inv, 0); TASSIGN(wm, C * C * sizeof(half));
    TASSIGN(um, (C * C + C * D) * sizeof(half));
    TileLeft<half, C, C> a; TileRight<half, C, D> b;
    TileAcc<float, C, D> result;
    TASSIGN(a, 0); TASSIGN(b, 0); TASSIGN(result, 0);
    for (uint32_t m = get_block_idx(); m < matrices; m += get_block_num()) {
        auto info = mk_solve::GetBSNDVarlenTileInfoFromCuSeqlens(m, H, C, cu);
        int valid = info.valid_size, head = m % H;
        int64_t pos = (info.bsnd_offset - head * C) / (H * C);
        InvTail it(valid, valid); Tail wt(valid, D), ut(valid, D);
        TASSIGN(it, 0); TASSIGN(wt, C * C * sizeof(half));
        TASSIGN(ut, (C * C + C * D) * sizeof(half));
        GlobalTensor<half, Shape2, Stride2> igm(inverse + info.bsnd_offset,
            {1, 1, 1, valid, valid}, {1, 1, 1, H * C, 1});
        GlobalTensor<half, Shape2, Stride2> wgm(w + (pos * H + head) * D,
            {1, 1, 1, valid, D}, {1, 1, 1, H * D, 1});
        GlobalTensor<half, Shape2, Stride2> ugm(u + (pos * H + head) * D,
            {1, 1, 1, valid, D}, {1, 1, 1, H * D, 1});
        TLOAD(it, igm); TLOAD(wt, wgm); TLOAD(ut, ugm); pipe_barrier(PIPE_ALL);
        if (valid != C) { TFILLPAD(it, it); TFILLPAD(wt, wt); TFILLPAD(ut, ut); }
        TMOV(a, inv); TMOV(b, um); pipe_barrier(PIPE_ALL);
        TMATMUL(result, a, b); pipe_barrier(PIPE_ALL);
        TileAcc<float, C, D, DYNAMIC, DYNAMIC> res(valid, D); TASSIGN(res, 0);
        TSTORE(ugm, res); pipe_barrier(PIPE_ALL);
        TMOV(b, wm); pipe_barrier(PIPE_ALL);
        TMATMUL(result, a, b); pipe_barrier(PIPE_ALL);
        TSTORE(wgm, res); pipe_barrier(PIPE_ALL);
    }
#endif
}
