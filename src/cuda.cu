inline __device__ int iterate(float re0, float im0, int count) {
    float re = re0;
    float im = im0;
    for (int i = 0; i < count; ++i) {
        if (re * re + im * im > 4.0) {
            return i;
        }
        float tmp = re * im;
        re = re * re - im * im + re0;
        im = tmp + tmp + im0;
    }
    return count;
}
extern "C" __global__ void kernel(
    int *out, float x0, float dx, float y0, float dy, int width, int count
) {
    float re0 = x0 + blockIdx.x * dx;
    float im0 = y0 + blockIdx.y * dy;
    out[blockIdx.y * width + blockIdx.x] = iterate(re0, im0, count);
}
