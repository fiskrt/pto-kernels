#if defined(GDN_PREFETCH) && !defined(GDN_NEGATIVE_A)
#error "GDN_PREFETCH requires in-place signed inversion (GDN_NEGATIVE_A)"
#endif

// Head-persistent pipeline. The two vector subblocks prepare chunk-local
// factors while the cube consumes earlier chunks. All inverse/WY/state/output
// intermediates stay in L1/L0; only final outputs are written by the cube.
template <int H, int C>
AICORE inline void mega_prepare_chunk(
    __gm__ half* q, __gm__ half* k, __gm__ half* v, __gm__ float* g, __gm__ half* beta,
    __gm__ half* raw_kk, __gm__ half* raw_qk, __gm__ float* lower, __gm__ float* full,
    __gm__ half* neg_i, __gm__ half* a, __gm__ half* w, __gm__ half* u,
    __gm__ half* scaled_k, __gm__ half* scaled_q, __gm__ half* attention, __gm__ half* decay,
    int64_t pos, int valid, int head, int kh, int64_t tokens, int64_t matrix, __gm__ volatile int32_t* control)
{
#if defined(__DAV_VEC__)
    constexpr int R = C / 2, F = R * C * sizeof(float);
    using S = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Str = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using VF = Tile<TileType::Vec, float, R, C, BLayout::RowMajor, R, C>;
    using VH = Tile<TileType::Vec, half, R, C, BLayout::RowMajor, R, C>;
    using HP = Tile<TileType::Vec, half, R, C, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using G = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, 1, C>;
    using GP = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, 1, DYNAMIC>;
    using Row = Tile<TileType::Vec, float, 1, R, BLayout::RowMajor, 1, R>;
    using Col = Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, R, 1>;
    using BH = Tile<TileType::Vec, half, 1, R, BLayout::RowMajor, 1, R>;
    using BP = Tile<TileType::Vec, half, 1, R, BLayout::RowMajor, 1, DYNAMIC>;
    VF values, keys, coeff, mask;
    VH data, raw;
    G gates;
    Row betas, row;
    BH beta_h;
    constexpr int GAddr = 5 * F, BAddr = GAddr + C * sizeof(float);
    constexpr int RAddr = BAddr + R * sizeof(float), BHAddr = RAddr + R * sizeof(float);
    TASSIGN(values, 0); TASSIGN(keys, F); TASSIGN(coeff, 2 * F); TASSIGN(mask, 3 * F);
    TASSIGN(data, 4 * F); TASSIGN(raw, 4 * F + F / 2);
    TASSIGN(gates, GAddr); TASSIGN(betas, BAddr); TASSIGN(row, RAddr); TASSIGN(beta_h, BHAddr);
    int stripe = get_subblockid() * R, count = min(R, valid - stripe);
    GP gp(valid); TASSIGN(gp, GAddr);
    GlobalTensor<float, S, Str> ggm(g + head * tokens + pos, {1, 1, 1, 1, valid}, {1, 1, 1, valid, 1});
    TLOAD(gp, ggm); pipe_barrier(PIPE_ALL);
    float last = gates.GetValue(valid - 1);
#ifdef GDN_ADAPTIVE_INVERSE
    int inverse_terms = 2;
#endif
    if (count > 0) {
        HP part(count, C), raw_part(count, C);
        TASSIGN(part, 4 * F); TASSIGN(raw_part, 4 * F + F / 2);
        BP bp(count); TASSIGN(bp, BHAddr);
        GlobalTensor<half, S, Str> bgm(beta + head * tokens + pos + stripe,
            {1, 1, 1, 1, count}, {1, 1, 1, count, 1});
        GlobalTensor<float, S, Str> fgm(full + stripe * C, {1, 1, 1, R, C}, {1, 1, 1, C, 1});
        TLOAD(bp, bgm); TLOAD(mask, fgm); pipe_barrier(PIPE_ALL);
        TCVT(betas, beta_h, RoundMode::CAST_NONE); PipeBarrierVec();
        Row grow; TASSIGN(grow, GAddr + stripe * sizeof(float));
        Col row_col, beta_col;
        TASSIGN(row_col, RAddr); TASSIGN(beta_col, BAddr);
        TMOV(row, grow); PipeBarrierVec();
        TCOLEXPAND(coeff, gates); PipeBarrierVec();
        TMULS(coeff, coeff, -1.0f); PipeBarrierVec();
        TROWEXPANDADD(coeff, coeff, row_col); PipeBarrierVec();
        TMINS(coeff, coeff, 0.0f); PipeBarrierVec();
        TEXP(coeff, coeff); PipeBarrierVec();
        TMUL(coeff, coeff, mask); PipeBarrierVec();
        int key_head = head / (H / kh);
        int64_t raw_offset = ((pos + stripe) * kh + key_head) * C;
        int64_t output_offset = (head * tokens + pos + stripe) * C;
        GlobalTensor<half, S, Str> qkgm(raw_qk + raw_offset, {1, 1, 1, count, C}, {1, 1, 1, kh * C, 1});
        TLOAD(raw_part, qkgm); pipe_barrier(PIPE_ALL);
        TCVT(values, raw, RoundMode::CAST_NONE); PipeBarrierVec();
        TMUL(values, values, coeff); PipeBarrierVec();
        TCVT(raw, values, RoundMode::CAST_NONE); pipe_barrier(PIPE_ALL);
        GlobalTensor<half, S, Str> attgm(attention + stripe * C,
            {1, 1, 1, count, C}, {1, 1, 1, C, 1});
        TSTORE(attgm, raw_part); pipe_barrier(PIPE_ALL);
        GlobalTensor<half, S, Str> kkgm(raw_kk + raw_offset, {1, 1, 1, count, C}, {1, 1, 1, kh * C, 1});
        GlobalTensor<float, S, Str> lgm(lower + stripe * C, {1, 1, 1, R, C}, {1, 1, 1, C, 1});
        TLOAD(raw_part, kkgm); TLOAD(mask, lgm); pipe_barrier(PIPE_ALL);
        TCVT(values, raw, RoundMode::CAST_NONE); PipeBarrierVec();
        TMUL(values, values, coeff); PipeBarrierVec();
        TMUL(values, values, mask); PipeBarrierVec();
        TROWEXPANDMUL(values, values, beta_col); PipeBarrierVec();
#ifdef GDN_NEGATIVE_A
        TMULS(values, values, -1.0f); PipeBarrierVec();
#endif
        TCVT(raw, values, RoundMode::CAST_NONE); pipe_barrier(PIPE_ALL);
        GlobalTensor<half, S, Str> agm(a + output_offset, {1, 1, 1, count, C}, {1, 1, 1, C, 1});
        TSTORE(agm, raw_part); pipe_barrier(PIPE_ALL);
#ifdef GDN_ADAPTIVE_INVERSE
        // Bound the geometric-series tail using the actual FP16 input matrix.
        // Each stripe bounds its diagonal block by the full row absolute sum.
        // The cube uses the larger bound for both diagonal blocks.
        TCVT(values, raw, RoundMode::CAST_NONE); PipeBarrierVec();
        TABS(values, values); PipeBarrierVec();
        Tile<TileType::Vec, float, R, C, BLayout::RowMajor, DYNAMIC, C> norm_values(count);
        Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, DYNAMIC, 1> sums(count);
        Tile<TileType::Vec, float, 1, R, BLayout::RowMajor, 1, DYNAMIC> sums_row(count);
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 1> norm;
        Row norm_tmp;
        TASSIGN(norm_values, 0); TASSIGN(sums, 3 * F); TASSIGN(sums_row, 3 * F);
        TASSIGN(norm, 3 * F + R * sizeof(float)); TASSIGN(norm_tmp, F);
        TROWSUM(sums, norm_values, keys); PipeBarrierVec();
        TROWMAX(norm, sums_row, norm_tmp); pipe_barrier(PIPE_ALL);
        float rho = norm.GetValue(0) * 1.001f;
        // For rho<1, omitted infinity norm <= rho^n/(1-rho).
        // These thresholds bound the diagonal truncation by 3.2e-6 or less;
        // the block join amplifies this by <1.5 at the loosest (n=8) case.
        inverse_terms = rho <= 0.0005f ? 2 : rho <= 0.03125f ? 4 :
                        rho <= 0.2f ? 8 : rho <= 0.4f ? 16 : rho <= 0.6f ? 32 : 64;
#endif

        GlobalTensor<half, S, Str> kgm(k + raw_offset, {1, 1, 1, count, C}, {1, 1, 1, kh * C, 1});
        TLOAD(part, kgm); pipe_barrier(PIPE_ALL);
        TCVT(keys, data, RoundMode::CAST_NONE);
        TEXP(row, grow); PipeBarrierVec();
        TMUL(row, row, betas); PipeBarrierVec();
        TMULS(row, row, -1.0f); PipeBarrierVec();
        TROWEXPANDMUL(values, keys, row_col); PipeBarrierVec();
        TCVT(data, values, RoundMode::CAST_NONE); pipe_barrier(PIPE_ALL);
        GlobalTensor<half, S, Str> wgm(w + output_offset, {1, 1, 1, count, C}, {1, 1, 1, C, 1});
        TSTORE(wgm, part); pipe_barrier(PIPE_ALL);
        TMULS(row, grow, -1.0f); PipeBarrierVec();
        TADDS(row, row, last); PipeBarrierVec();
        TEXP(row, row); PipeBarrierVec();
        TROWEXPANDMUL(values, keys, row_col); PipeBarrierVec();
        TCVT(data, values, RoundMode::CAST_NONE); pipe_barrier(PIPE_ALL);
        GlobalTensor<half, S, Str> ktgm(scaled_k + output_offset,
            {1, 1, 1, count, C}, {1, 1, 1, C, 1});
        TSTORE(ktgm, part); pipe_barrier(PIPE_ALL);
        GlobalTensor<half, S, Str> vgm(v + ((pos + stripe) * H + head) * C, {1, 1, 1, count, C}, {1, 1, 1, H * C, 1});
        TLOAD(part, vgm); pipe_barrier(PIPE_ALL);
        TCVT(values, data, RoundMode::CAST_NONE); PipeBarrierVec();
        TROWEXPANDMUL(values, values, beta_col); PipeBarrierVec();
        TCVT(data, values, RoundMode::CAST_NONE); pipe_barrier(PIPE_ALL);
        GlobalTensor<half, S, Str> ugm(u + output_offset, {1, 1, 1, count, C}, {1, 1, 1, C, 1});
        TSTORE(ugm, part); pipe_barrier(PIPE_ALL);
        GlobalTensor<half, S, Str> qgm(q + raw_offset, {1, 1, 1, count, C}, {1, 1, 1, kh * C, 1});
        TLOAD(part, qgm); pipe_barrier(PIPE_ALL);
        TCVT(values, data, RoundMode::CAST_NONE);
        TEXP(row, grow); PipeBarrierVec();
        TROWEXPANDMUL(values, values, row_col); PipeBarrierVec();
        TCVT(data, values, RoundMode::CAST_NONE); pipe_barrier(PIPE_ALL);
        GlobalTensor<half, S, Str> qout(scaled_q + output_offset,
            {1, 1, 1, count, C}, {1, 1, 1, C, 1});
        TSTORE(qout, part); pipe_barrier(PIPE_ALL);
    }
    GlobalTensor<half, S, Str> igm(neg_i + stripe * C, {1, 1, 1, R, C}, {1, 1, 1, C, 1});
    TLOAD(data, igm); pipe_barrier(PIPE_ALL);
    TCVT(values, data, RoundMode::CAST_NONE);
    TEXP(gates, gates); pipe_barrier(PIPE_ALL);
    float exp_last = gates.GetValue(valid - 1);
    TMULS(values, values, -exp_last); PipeBarrierVec();
    TCVT(data, values, RoundMode::CAST_NONE); pipe_barrier(PIPE_ALL);
    GlobalTensor<half, S, Str> dgm(decay + stripe * C,
        {1, 1, 1, R, C}, {1, 1, 1, C, 1});
    TSTORE(dgm, data); pipe_barrier(PIPE_ALL);
#ifdef GDN_ADAPTIVE_INVERSE
    control[0] = inverse_terms;
    dcci(control, SINGLE_CACHE_LINE);
    // Publish scalar-written metadata before the MTE3 ready notification.
    dsb(DSB_DDR);
#endif

#endif
}

template <int H, int C>
AICORE inline void mega_fused_pipeline(
    __gm__ half* q, __gm__ half* k, __gm__ half* v, __gm__ float* g, __gm__ half* beta,
    __gm__ half* raw_kk, __gm__ half* raw_qk, __gm__ float* lower, __gm__ float* full,
    __gm__ half* neg_i, __gm__ half* a_gm, __gm__ half* w, __gm__ half* u,
    __gm__ half* scaled_k, __gm__ half* scaled_q, __gm__ half* output, __gm__ half* decays,
    __gm__ half* final_state, __gm__ half* initial, bool has_initial,
    __gm__ int32_t* cu, int64_t sequences, int kh, int64_t tokens, __gm__ half* debug)
{
#ifdef GDN_FUSED_TIMING
    uint64_t wait_ticks = 0, phase_ticks = 0, rest_ticks = 0, chunks_done = 0, stamp = 0;
#endif
#if defined(__DAV_VEC__)
    set_mask_norm(); set_vector_mask(-1, -1);
#endif
#if defined(__DAV_CUBE__)
    static_assert(C == 128);
    constexpr int Bytes = C * C * sizeof(half);
    using Mat = Tile<TileType::Mat, half, C, C, BLayout::ColMajor, C, C, SLayout::RowMajor, 512>;
    using Tail = Tile<TileType::Mat, half, C, C, BLayout::ColMajor, DYNAMIC, DYNAMIC,
                      SLayout::RowMajor, 512, PadValue::Zero>;
    using Trans = Tile<TileType::Mat, half, C, C, BLayout::RowMajor, C, C, SLayout::ColMajor, 512>;
    Mat input, negative, inverse, negative_identity, identity;
    Mat state, kt, query, attention, decay, residual;
    TASSIGN(input, 0); TASSIGN(negative, Bytes); TASSIGN(inverse, 2 * Bytes);
    TASSIGN(negative_identity, 3 * Bytes); TASSIGN(identity, 4 * Bytes);
    // Inverse helper scratch occupies [5*Bytes, 6*Bytes).
    TASSIGN(state, 8 * Bytes); TASSIGN(kt, 9 * Bytes); TASSIGN(query, 10 * Bytes);
    TASSIGN(attention, 11 * Bytes); TASSIGN(decay, 12 * Bytes); TASSIGN(residual, 13 * Bytes);
    TileLeft<half, C, C> left; TileRight<half, C, C> right;
    TileAcc<float, C, C> acc, out_acc;
    TASSIGN(left, 0); TASSIGN(right, 0); TASSIGN(acc, 0); TASSIGN(out_acc, C * C * sizeof(float));
    mega_load_packed<C, C, C>(negative_identity, negative_identity, neg_i, C, C);
    pipe_barrier(PIPE_ALL);
    TMOV(left, negative_identity); TMOV(right, negative_identity); pipe_barrier(PIPE_ALL);
    TMATMUL(acc, left, right); pipe_barrier(PIPE_ALL);
    TMOV(identity, acc); pipe_barrier(PIPE_ALL);
#endif
    for (int64_t work = get_block_idx(); work < sequences * H; work += get_block_num()) {
        int seq = work / H, head = work % H;
        int64_t start = cu[seq], end = cu[seq + 1], chunk_base = 0;
        for (int s = 0; s < seq; ++s) chunk_base += (cu[s + 1] - cu[s] + C - 1) / C;
#if defined(__DAV_CUBE__)
        if (has_initial) {
            mega_load_packed<C, C, C>(state, state, initial + work * C * C, C, C);
            pipe_barrier(PIPE_ALL);
            // Nonempty sequences overwrite this accumulator before using it.
            if (start == end) {
                TMOV(left, state); TMOV(right, identity); pipe_barrier(PIPE_ALL);
                TMATMUL(acc, left, right); pipe_barrier(PIPE_ALL);
            }
        } else {
            TMOV(left, identity); TMOV(right, identity); pipe_barrier(PIPE_ALL);
            TMATMUL(acc, left, right); pipe_barrier(PIPE_ALL);
            TMOV(left, negative_identity); pipe_barrier(PIPE_ALL);
            TMATMUL_ACC(acc, acc, left, right); pipe_barrier(PIPE_ALL);
            TMOV(state, acc); pipe_barrier(PIPE_ALL);
        }
#if __CCE_AICORE__ == 220
        ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (7 << 8));
        ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (8 << 8));
#else
        GdnSignalBothVec<PIPE_FIX>(7); GdnSignalBothVec<PIPE_FIX>(8);
#endif
#endif
        int ci = 0;
        for (int64_t pos = start; pos < end; pos += C, ++ci) {
            int valid = min(static_cast<int64_t>(C), end - pos);
            int slot = ci & 1;
            int64_t m = (chunk_base + ci) * H + head;
#if defined(__DAV_VEC__)
#ifdef GDN_FUSED_TIMING
            stamp = get_sys_cnt();
#endif
#if __CCE_AICORE__ == 220
            wait_flag_dev(7 + slot);
#else
            wait_intra_block(PIPE_MTE3, 7 + slot);
#endif
#ifdef GDN_FUSED_TIMING
            wait_ticks += get_sys_cnt() - stamp;
            stamp = get_sys_cnt();
#endif
            auto slot_buffer = decays + (get_block_idx() * 4 + slot * 2) * C * C;
            mega_prepare_chunk<H, C>(q, k, v, g, beta, raw_kk, raw_qk, lower, full, neg_i,
                                     a_gm, w, u, scaled_k, scaled_q, slot_buffer, slot_buffer + C * C,
                                     pos, valid, head, kh, tokens, m,
                                     reinterpret_cast<__gm__ int32_t*>(debug) + get_block_idx() * 16384 +
                                         8192 + slot * 64 + get_subblockid() * 32);
#ifdef GDN_FUSED_TIMING
            phase_ticks += get_sys_cnt() - stamp;
            ++chunks_done;
#endif
#if __CCE_AICORE__ == 220
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | ((5 + slot) << 8));
#else
            set_intra_block(PIPE_MTE3, 5 + slot);
#endif
#endif
#if defined(__DAV_CUBE__)
#ifdef GDN_FUSED_TIMING
            stamp = get_sys_cnt();
#endif
#if __CCE_AICORE__ == 220
            wait_flag_dev(5 + slot);

#else
            GdnWaitBothVec<PIPE_MTE2>(5 + slot);

#endif
#ifdef GDN_FUSED_TIMING
            wait_ticks += get_sys_cnt() - stamp;
            stamp = get_sys_cnt();
#endif
            int64_t offset = (pos * H + head) * C;
            int64_t packed = (head * tokens + pos) * C;
            auto slot_buffer = decays + (get_block_idx() * 4 + slot * 2) * C * C;
#ifdef GDN_NEGATIVE_A
            Tail at(valid, valid); TASSIGN(at, 2 * Bytes);
            mega_load_packed<C, C, C>(inverse, at, a_gm + packed, valid, valid);
            pipe_barrier(PIPE_ALL);
            if (valid != C) TFILLPAD(at, at);
#else
            Tail at(valid, valid); TASSIGN(at, 0);
            mega_load_packed<C, C, C>(input, at, a_gm + packed, valid, valid);
            pipe_barrier(PIPE_ALL);
            if (valid != C) TFILLPAD(at, at);
            TMOV(left, negative_identity); TMOV(right, input); pipe_barrier(PIPE_ALL);
            TMATMUL(acc, left, right); pipe_barrier(PIPE_ALL);
            TMOV(negative, acc); TMOV(inverse, acc); pipe_barrier(PIPE_ALL);
#endif
            int terms = 64;
#ifdef GDN_ADAPTIVE_INVERSE
            auto controls = reinterpret_cast<__gm__ volatile int32_t*>(debug) + get_block_idx() * 16384 + 8192 + slot * 64;
            dcci(controls, SINGLE_CACHE_LINE); dcci(controls + 32, SINGLE_CACHE_LINE);
            dsb(DSB_DDR);
            int first_terms = controls[0], second_terms = controls[32];
            terms = max(first_terms, second_terms);
#endif
            Tail wt(valid, C), ut(valid, C), ktail(valid, C), qtail(valid, C), att(valid, valid);
            TASSIGN(wt, 0); TASSIGN(ut, Bytes); TASSIGN(ktail, 9 * Bytes);
            TASSIGN(qtail, 10 * Bytes); TASSIGN(att, 11 * Bytes);
#ifdef GDN_PREFETCH
            mega_load_packed<C, C, C>(input, wt, w + packed, valid, C);
            mega_load_packed<C, C, C>(negative, ut, u + packed, valid, C);
            mega_load_packed<C, C, C>(kt, ktail, scaled_k + packed, valid, C);
            mega_load_packed<C, C, C>(query, qtail, scaled_q + packed, valid, C);
            mega_load_packed<C, C, C>(attention, att, slot_buffer, valid, valid);
            mega_load_packed<C, C, C>(decay, decay, slot_buffer + C * C, C, C);
#endif
#ifdef GDN_NEGATIVE_A
            // With B=-A and +I, the same polynomial computes (I+A)^-1.
            // Invert diagonal blocks in place; the lower offdiagonal B is
            // untouched until the join consumes and replaces it.
            mega_inverse_diagonal<64>(inverse, identity, inverse, 0, terms);
            mega_inverse_diagonal<64>(inverse, identity, inverse, 64, terms);
            mega_inverse_join<64>(inverse, inverse, 0);
#else
            mega_inverse_diagonal<64>(input, negative_identity, inverse, 0, terms);
            mega_inverse_diagonal<64>(input, negative_identity, inverse, 64, terms);
            mega_inverse_join<64>(negative, inverse, 0);
#endif
#ifdef GDN_FUSED_TIMING
            phase_ticks += get_sys_cnt() - stamp;
            stamp = get_sys_cnt();
#endif
#ifndef GDN_PREFETCH
            mega_load_packed<C, C, C>(input, wt, w + packed, valid, C);
            mega_load_packed<C, C, C>(negative, ut, u + packed, valid, C);
            mega_load_packed<C, C, C>(kt, ktail, scaled_k + packed, valid, C);
            mega_load_packed<C, C, C>(query, qtail, scaled_q + packed, valid, C);
            mega_load_packed<C, C, C>(attention, att, slot_buffer, valid, valid);
            mega_load_packed<C, C, C>(decay, decay, slot_buffer + C * C, C, C);
#endif
            pipe_barrier(PIPE_ALL);
            if (valid != C) {
                TFILLPAD(wt, wt); TFILLPAD(ut, ut); TFILLPAD(ktail, ktail);
                TFILLPAD(qtail, qtail); TFILLPAD(att, att);
            }
            // Both slot matrices have reached L1; the producer may reuse them.
#if __CCE_AICORE__ == 220
            ffts_cross_core_sync(PIPE_MTE2, 1 | (2 << 4) | ((7 + slot) << 8));
#else
            GdnSignalBothVec<PIPE_MTE2>(7 + slot);
#endif
            TMOV(left, inverse); TMOV(right, negative); pipe_barrier(PIPE_ALL);
            TMATMUL(acc, left, right); pipe_barrier(PIPE_ALL); // U, retained in FP32.
            TMOV(right, input); pipe_barrier(PIPE_ALL);
            TMATMUL(out_acc, left, right); pipe_barrier(PIPE_ALL);

            TMOV(input, out_acc); pipe_barrier(PIPE_ALL); // -W.
            if (ci != 0 || has_initial) {
                TMOV(left, input); TMOV(right, state); pipe_barrier(PIPE_ALL);
                TMATMUL_ACC(acc, acc, left, right); pipe_barrier(PIPE_ALL);
            }
            TMOV(residual, acc); pipe_barrier(PIPE_ALL);
#ifdef GDN_JOINT_PRODUCTS
            // Output and next state consume exactly the same two RHS matrices.
            // [O; S_next] = [Q_scaled; decay*I] S + [attention; K_tilde^T] R.
            TileLeft<half, 2 * C, C> output_left;
            TileLeft<half, C, C> state_left;
            TileAcc<float, 2 * C, C> output_state;
            TASSIGN(output_left, 0); TASSIGN(state_left, Bytes); TASSIGN(output_state, 0);
            TMOV(left, query); TMOV(state_left, decay); TMOV(right, state);
            pipe_barrier(PIPE_ALL);
            TMATMUL(output_state, output_left, right); pipe_barrier(PIPE_ALL);
            Trans kt_trans; TRESHAPE(kt_trans, kt);
            TMOV(left, attention); TEXTRACT(state_left, kt_trans, 0, 0); TMOV(right, residual);
            pipe_barrier(PIPE_ALL);
            TMATMUL_ACC(output_state, output_state, output_left, right); pipe_barrier(PIPE_ALL);
            TEXTRACT(state, output_state, C, 0); pipe_barrier(PIPE_ALL);
            TileAcc<float, 2 * C, C, DYNAMIC, C> ot(valid);
            TASSIGN(ot, 0);
            mega_store_packed<C, C, H * C>(ot, ot, output + offset, valid, C);
            pipe_barrier(PIPE_ALL);
#else
            TMOV(left, query); TMOV(right, state); pipe_barrier(PIPE_ALL);
            TMATMUL(out_acc, left, right); pipe_barrier(PIPE_ALL);
            TMOV(left, state); TMOV(right, decay); pipe_barrier(PIPE_ALL);
            TMATMUL(acc, left, right); pipe_barrier(PIPE_ALL);
            Trans kt_trans; TRESHAPE(kt_trans, kt);
            TEXTRACT(left, kt_trans, 0, 0); TMOV(right, residual); pipe_barrier(PIPE_ALL);
            TMATMUL_ACC(acc, acc, left, right); pipe_barrier(PIPE_ALL);
            TMOV(state, acc); pipe_barrier(PIPE_ALL);
            TMOV(left, attention); pipe_barrier(PIPE_ALL);
            TMATMUL_ACC(out_acc, out_acc, left, right); pipe_barrier(PIPE_ALL);
            TileAcc<float, C, C, DYNAMIC, DYNAMIC> ot(valid, C);
            TASSIGN(ot, C * C * sizeof(float));
            mega_store_packed<C, C, H * C>(out_acc, ot, output + offset, valid, C);
            pipe_barrier(PIPE_ALL);
#endif
#ifdef GDN_FUSED_TIMING
            rest_ticks += get_sys_cnt() - stamp;
            ++chunks_done;
#endif
#endif
        }
#if defined(__DAV_CUBE__)
#ifdef GDN_JOINT_PRODUCTS
        if (end > start) {
            TileAcc<float, 2 * C, C, C, C> last_state;
            TASSIGN(last_state, C * 16 * sizeof(float));
            mega_store_packed<C, C, C>(last_state, last_state, final_state + work * C * C, C, C);
        } else
#endif
        {
        mega_store_packed<C, C, C>(acc, acc, final_state + work * C * C, C, C);
        }
        pipe_barrier(PIPE_ALL);
#elif defined(__DAV_VEC__)
        // Two initial credits plus one replenishment per consumed chunk leave
        // exactly two free-slot signals to drain before reusing these flags.
#if __CCE_AICORE__ == 220
        wait_flag_dev(7); wait_flag_dev(8);
#else
        wait_intra_block(PIPE_MTE3, 7); wait_intra_block(PIPE_MTE3, 8);
#endif
#endif
    }
#ifdef GDN_FUSED_TIMING
#if defined(__DAV_CUBE__)
    int unit = 0;
#else
    int unit = 1 + get_subblockid();
#endif
    auto timing = reinterpret_cast<__gm__ uint64_t*>(debug) + (get_block_idx() * 3 + unit) * 16;
    timing[0] = wait_ticks; timing[1] = phase_ticks; timing[2] = rest_ticks; timing[3] = chunks_done;
    dcci(timing, SINGLE_CACHE_LINE);
#endif

}
