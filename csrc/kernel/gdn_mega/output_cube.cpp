// QK is shared across grouped heads; vector preparation applies its gate and
// scales Q while the recurrent state runs. Finish both output terms in FP32.
template <int H, int C, int D>
AICORE inline void mega_output_cube(__gm__ half* query, __gm__ half* attention,
                                   __gm__ half* values, __gm__ half* states, __gm__ half* out,
                                   __gm__ int32_t* cu, uint32_t matrices)
{
#if defined(__DAV_CUBE__)
    static_assert(C == D);
    constexpr int Bytes = C * D * sizeof(half);
    using Mat = Tile<TileType::Mat, half, C, D, BLayout::ColMajor, C, D, SLayout::RowMajor, 512>;
    using Tail = Tile<TileType::Mat, half, C, D, BLayout::ColMajor, DYNAMIC, DYNAMIC,
                      SLayout::RowMajor, 512, PadValue::Zero>;
    using Shape2 = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Stride2 = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    Mat q, a, v, s;
    TASSIGN(q, 0); TASSIGN(a, Bytes); TASSIGN(v, 2 * Bytes); TASSIGN(s, 3 * Bytes);
    TileLeft<half, C, D> left; TileRight<half, D, D> right;
    TileAcc<float, C, D> result;
    TASSIGN(left, 0); TASSIGN(right, 0); TASSIGN(result, 0);
    for (uint32_t m = get_block_idx(); m < matrices; m += get_block_num()) {
        auto info = mk_solve::GetBSNDVarlenTileInfoFromCuSeqlens(m, H, C, cu);
        int valid = info.valid_size;
        Tail qt(valid, D), at(valid, valid), vt(valid, D);
        TASSIGN(qt, 0); TASSIGN(at, Bytes); TASSIGN(vt, 2 * Bytes);
        GlobalTensor<half, Shape2, Stride2> qgm(query + info.bsnd_offset,
            {1, 1, 1, valid, D}, {1, 1, 1, H * D, 1});
        GlobalTensor<half, Shape2, Stride2> agm(attention + info.bsnd_offset,
            {1, 1, 1, valid, valid}, {1, 1, 1, H * C, 1});
        GlobalTensor<half, Shape2, Stride2> vgm(values + info.bsnd_offset,
            {1, 1, 1, valid, D}, {1, 1, 1, H * D, 1});
        GlobalTensor<half, Shape2, Stride2> sgm(states + m * D * D,
            {1, 1, 1, D, D}, {1, 1, 1, D, 1});
        mega_load_packed<C, D, H * D>(q, qt, query + info.bsnd_offset, valid, D);
        mega_load_packed<C, C, H * C>(a, at, attention + info.bsnd_offset, valid, valid);
        mega_load_packed<C, D, H * D>(v, vt, values + info.bsnd_offset, valid, D);
        mega_load_packed<D, D, D>(s, s, states + m * D * D, D, D);
        pipe_barrier(PIPE_ALL);
        if (valid != C) { TFILLPAD(qt, qt); TFILLPAD(at, at); TFILLPAD(vt, vt); }
        TMOV(left, q); TMOV(right, s); pipe_barrier(PIPE_ALL);
        TMATMUL(result, left, right); pipe_barrier(PIPE_ALL);
        TMOV(left, a); TMOV(right, v); pipe_barrier(PIPE_ALL);
        TMATMUL_ACC(result, result, left, right); pipe_barrier(PIPE_ALL);
        TileAcc<float, C, D, DYNAMIC, DYNAMIC> tail(valid, D); TASSIGN(tail, 0);
        GlobalTensor<half, Shape2, Stride2> ogm(out + info.bsnd_offset,
            {1, 1, 1, valid, D}, {1, 1, 1, H * D, 1});
        mega_store_packed<C, D, H * D>(result, tail, out + info.bsnd_offset, valid, D);
        pipe_barrier(PIPE_ALL);
    }
#endif
}
