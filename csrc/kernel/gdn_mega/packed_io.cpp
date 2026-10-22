// Static full-tile descriptors avoid dynamic DMA setup on the common path.
template <int R, int C, int Stride, typename Full, typename Tail>
AICORE inline void mega_load_packed(Full& full, Tail& tail, __gm__ half* pointer, int rows, int cols)
{
    if (rows == R && cols == C) {
        GlobalTensor<half, Shape<1, 1, 1, R, C>, pto::Stride<1, 1, 1, Stride, 1>> gm(pointer);
        TLOAD(full, gm);
    } else {
        GlobalTensor<half, Shape<1, 1, 1, DYNAMIC, DYNAMIC>, pto::Stride<1, 1, 1, Stride, 1>> gm(
            pointer, {1, 1, 1, rows, cols});
        TLOAD(tail, gm);
    }
}

template <int R, int C, int Stride, typename Full, typename Tail>
AICORE inline void mega_store_packed(Full& full, Tail& tail, __gm__ half* pointer, int rows, int cols)
{
    if (rows == R && cols == C) {
        GlobalTensor<half, Shape<1, 1, 1, R, C>, pto::Stride<1, 1, 1, Stride, 1>> gm(pointer);
        TSTORE(gm, full);
    } else {
        GlobalTensor<half, Shape<1, 1, 1, DYNAMIC, DYNAMIC>, pto::Stride<1, 1, 1, Stride, 1>> gm(
            pointer, {1, 1, 1, rows, cols});
        TSTORE(gm, tail);
    }
}
