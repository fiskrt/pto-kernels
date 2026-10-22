// Prepare chunk-local factors while the cube cores invert the triangular tiles.
template <int H, int C, int D>
AICORE inline void mega_prepare_state(__gm__ half* k, __gm__ float* g, __gm__ half* neg_i,
                                     __gm__ half* scaled_k, __gm__ half* decay,
                                     __gm__ int32_t* cu, uint32_t matrices, int kh, int64_t tokens, uint32_t first = UINT32_MAX, uint32_t last = UINT32_MAX)
{
#if defined(__DAV_VEC__)
    static_assert(C == D);
    set_mask_norm(); set_vector_mask(-1, -1);
    constexpr int R = C / 2, F = R * D * sizeof(float);
    using Shape2 = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Stride2 = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using Half = Tile<TileType::Vec, half, R, D, BLayout::RowMajor, R, D>;
    using Part = Tile<TileType::Vec, half, R, D, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using Float = Tile<TileType::Vec, float, R, D, BLayout::RowMajor, R, D>;
    using Gate = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, 1, C>;
    using GatePart = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, 1, DYNAMIC>;
    using Row = Tile<TileType::Vec, float, 1, R, BLayout::RowMajor, 1, R>;
    using Col = Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, R, 1>;
    Half half_tile, identity_h;
    Float values, coeff, identity_f;
    Gate gates;
    Row row;
    TASSIGN(half_tile, 0); TASSIGN(identity_h, F / 2);
    TASSIGN(values, F); TASSIGN(coeff, 2 * F); TASSIGN(identity_f, 3 * F);
    TASSIGN(gates, 4 * F); TASSIGN(row, 4 * F + C * sizeof(float));
    int stripe = get_subblockid() * R;
    GlobalTensor<half, Shape2, Stride2> id_gm(neg_i + stripe * D, {1, 1, 1, R, D}, {1, 1, 1, D, 1});
    TLOAD(identity_h, id_gm); pipe_barrier(PIPE_ALL);
    TCVT(identity_f, identity_h, RoundMode::CAST_NONE); PipeBarrierVec();
    for (uint32_t m = first == UINT32_MAX ? get_block_idx() : first;
         m < (last == UINT32_MAX ? matrices : last);
         m += first == UINT32_MAX ? get_block_num() : matrices) {
        auto info = mk_solve::GetBSNDVarlenTileInfoFromCuSeqlens(m, H, C, cu);
        int head = m % H, valid = info.valid_size;
        int count = min(R, valid - stripe);
        int64_t pos = (info.bsnd_offset - head * C) / (H * C);
        GatePart gp(valid); TASSIGN(gp, 4 * F);
        GlobalTensor<float, Shape2, Stride2> ggm(g + head * tokens + pos,
            {1, 1, 1, 1, valid}, {1, 1, 1, valid, 1});
        TLOAD(gp, ggm); pipe_barrier(PIPE_ALL);
        float last = gates.GetValue(valid - 1);
        if (count > 0) {
            Part kp(count, D); TASSIGN(kp, 0);
            GlobalTensor<half, Shape2, Stride2> kgm(k + ((pos + stripe) * kh + head / (H / kh)) * D,
                {1, 1, 1, count, D}, {1, 1, 1, kh * D, 1});
            TLOAD(kp, kgm); pipe_barrier(PIPE_ALL);
            Row sub; TASSIGN(sub, 4 * F + stripe * sizeof(float));
            TMULS(row, sub, -1.0f); PipeBarrierVec();
            TADDS(row, row, last); PipeBarrierVec();
            TEXP(row, row); PipeBarrierVec();
            Col col; TASSIGN(col, 4 * F + C * sizeof(float));
            TROWEXPAND(coeff, col);
            TCVT(values, half_tile, RoundMode::CAST_NONE); PipeBarrierVec();
            TMUL(values, values, coeff); PipeBarrierVec();
            TCVT(half_tile, values, RoundMode::CAST_NONE); pipe_barrier(PIPE_ALL);
            GlobalTensor<half, Shape2, Stride2> output(scaled_k + ((pos + stripe) * H + head) * D,
                {1, 1, 1, count, D}, {1, 1, 1, H * D, 1});
            TSTORE(output, kp); pipe_barrier(PIPE_ALL);
        }
        TEXP(gates, gates); pipe_barrier(PIPE_ALL);
        float exp_last = gates.GetValue(valid - 1);
        TMULS(values, identity_f, -exp_last); PipeBarrierVec();
        TCVT(half_tile, values, RoundMode::CAST_NONE); pipe_barrier(PIPE_ALL);
        GlobalTensor<half, Shape2, Stride2> decay_gm(decay + m * D * D + stripe * D,
            {1, 1, 1, R, D}, {1, 1, 1, D, 1});
        TSTORE(decay_gm, half_tile); pipe_barrier(PIPE_ALL);
    }
#endif
}

// The state stays in L1 between chunks. Residual subtraction and decay are
// expressed as matrix accumulations, removing all recurrent cube/vector GM
// roundtrips. W is prepared with a negative sign by the WY input stage.
template <int H, int C, int D>
AICORE inline void mega_state_cube(__gm__ half* scaled_k, __gm__ half* negative_w, __gm__ half* u,
                                  __gm__ half* states, __gm__ half* vnew, __gm__ half* final_state,
                                  __gm__ half* initial, bool has_initial, __gm__ half* neg_i,
                                  __gm__ int32_t* cu, int64_t sequences)
{
#if defined(__DAV_CUBE__)
    static_assert(C == D);
    constexpr int Bytes = D * D * sizeof(half);
    using Mat = Tile<TileType::Mat, half, D, D, BLayout::ColMajor, D, D, SLayout::RowMajor, 512>;
    using Tail = Tile<TileType::Mat, half, D, D, BLayout::ColMajor, DYNAMIC, DYNAMIC,
                      SLayout::RowMajor, 512, PadValue::Zero>;
    using Trans = Tile<TileType::Mat, half, D, D, BLayout::RowMajor, D, D, SLayout::ColMajor, 512>;
    using Shape2 = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Stride2 = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    Mat state, wm, um, km, residual, decay, identity, negative_identity;
    TASSIGN(state, 0); TASSIGN(wm, Bytes); TASSIGN(um, 2 * Bytes); TASSIGN(km, 3 * Bytes);
    TASSIGN(residual, 4 * Bytes); TASSIGN(decay, 5 * Bytes);
    TASSIGN(identity, 6 * Bytes); TASSIGN(negative_identity, 7 * Bytes);
    TileLeft<half, D, D> a; TileRight<half, D, D> b;
    TileAcc<float, D, D> sc, vc;
    TASSIGN(a, 0); TASSIGN(b, 0); TASSIGN(sc, 0); TASSIGN(vc, D * D * sizeof(float));
    GlobalTensor<half, Shape2, Stride2> idgm(neg_i, {1, 1, 1, D, D}, {1, 1, 1, D, 1});
    TLOAD(negative_identity, idgm); pipe_barrier(PIPE_ALL);
    TMOV(a, negative_identity); TMOV(b, negative_identity); pipe_barrier(PIPE_ALL);
    TMATMUL(sc, a, b); pipe_barrier(PIPE_ALL);
    TMOV(identity, sc); pipe_barrier(PIPE_ALL);
    for (int64_t work = get_block_idx(); work < sequences * H; work += get_block_num()) {
        int head = work % H, seq = work / H;
        int64_t start = cu[seq], end = cu[seq + 1], chunk_base = 0;
        for (int i = 0; i < seq; ++i) chunk_base += (cu[i + 1] - cu[i] + C - 1) / C;
        if (has_initial) {
            GlobalTensor<half, Shape2, Stride2> hgm(initial + work * D * D,
                {1, 1, 1, D, D}, {1, 1, 1, D, 1});
            TLOAD(state, hgm); pipe_barrier(PIPE_ALL);
            TMOV(a, state); TMOV(b, identity); pipe_barrier(PIPE_ALL);
            TMATMUL(sc, a, b); pipe_barrier(PIPE_ALL);
        } else {
            TMOV(a, identity); TMOV(b, identity); pipe_barrier(PIPE_ALL);
            TMATMUL(sc, a, b); pipe_barrier(PIPE_ALL);
            TMOV(a, negative_identity); pipe_barrier(PIPE_ALL);
            TMATMUL_ACC(sc, sc, a, b); pipe_barrier(PIPE_ALL);
            TMOV(state, sc); pipe_barrier(PIPE_ALL);
        }
        int ci = 0;
        for (int64_t pos = start; pos < end; pos += C, ++ci) {
            int valid = min(static_cast<int64_t>(C), end - pos);
            int64_t offset = (pos * H + head) * D;
            Tail wt(valid, D), ut(valid, D), kt(valid, D);
            TASSIGN(wt, Bytes); TASSIGN(ut, 2 * Bytes); TASSIGN(kt, 3 * Bytes);
            GlobalTensor<half, Shape2, Stride2> wgm(negative_w + offset,
                {1, 1, 1, valid, D}, {1, 1, 1, H * D, 1});
            GlobalTensor<half, Shape2, Stride2> ugm(u + offset,
                {1, 1, 1, valid, D}, {1, 1, 1, H * D, 1});
            GlobalTensor<half, Shape2, Stride2> kgm(scaled_k + offset,
                {1, 1, 1, valid, D}, {1, 1, 1, H * D, 1});
            GlobalTensor<half, Shape2, Stride2> sgm(states + ((chunk_base + ci) * H + head) * D * D,
                {1, 1, 1, D, D}, {1, 1, 1, D, 1});
            TLOAD(wt, wgm); TLOAD(ut, ugm); TLOAD(kt, kgm); TLOAD(decay, sgm);
            pipe_barrier(PIPE_ALL);
            if (valid != C) { TFILLPAD(wt, wt); TFILLPAD(ut, ut); TFILLPAD(kt, kt); }
            TSTORE(sgm, sc); pipe_barrier(PIPE_ALL);
            TMOV(a, um); TMOV(b, identity); pipe_barrier(PIPE_ALL);
            TMATMUL(vc, a, b); pipe_barrier(PIPE_ALL);
            if (ci != 0 || has_initial) {
                TMOV(a, wm); TMOV(b, state); pipe_barrier(PIPE_ALL);
                TMATMUL_ACC(vc, vc, a, b); pipe_barrier(PIPE_ALL);
            }
            TMOV(residual, vc);
            TileAcc<float, D, D, DYNAMIC, DYNAMIC> vr(valid, D); TASSIGN(vr, D * D * sizeof(float));
            GlobalTensor<half, Shape2, Stride2> vgm(vnew + offset,
                {1, 1, 1, valid, D}, {1, 1, 1, H * D, 1});
            TSTORE(vgm, vr); pipe_barrier(PIPE_ALL);
            TMOV(a, state); TMOV(b, decay); pipe_barrier(PIPE_ALL);
            TMATMUL(sc, a, b); pipe_barrier(PIPE_ALL);
            Trans ktm; TRESHAPE(ktm, km);
            TEXTRACT(a, ktm, 0, 0); TMOV(b, residual); pipe_barrier(PIPE_ALL);
            TMATMUL_ACC(sc, sc, a, b); pipe_barrier(PIPE_ALL);
            TMOV(state, sc); pipe_barrier(PIPE_ALL);
        }
        GlobalTensor<half, Shape2, Stride2> fgm(final_state + work * D * D,
            {1, 1, 1, D, D}, {1, 1, 1, D, 1});
        TSTORE(fgm, sc); pipe_barrier(PIPE_ALL);
    }
#endif
}
