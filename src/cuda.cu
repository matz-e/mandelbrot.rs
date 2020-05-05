extern "C" __global__ void kernel(const float *re, const float *im, int *out, int size, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float new_re = re[i];
        float new_im = im[i];
        float re_sqr = new_re * new_re;
        float im_sqr = new_im * new_im;
        out[i] = 0;
        for (; out[i] < count && re_sqr + im_sqr <= 4.0; out[i] += 1) {
            float tmp = new_re * new_im;
            new_re = re_sqr - im_sqr + re[i];
            new_im = tmp + tmp + im[i];
            re_sqr = new_re * new_re;
            im_sqr = new_im * new_im;
        }
    }
}
