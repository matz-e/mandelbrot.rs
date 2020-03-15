extern crate packed_simd;

use packed_simd::*;

fn kernel(re: f32x8, im: f32x8, count: i32) -> i32x8 {
    let r2 = f32x8::splat(4.0);
    let mut counts = i32x8::splat(0);
    let mut z_re = re;
    let mut z_im = im;
    for _i in 0..count {
        let m = (z_re * z_re + z_im * z_im).le(r2);
        counts = m.select(counts + 1, counts);
        let new_re = z_re * z_re - z_im * z_im + re;
        let new_im = 2.0 * z_re * z_im + im;
        z_re = new_re;
        z_im = new_im;
        if m.none() {
            break;
        }
    }
    counts
}

pub fn mandelbrot(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    width: i32,
    height: i32,
    count: i32,
    output: &mut [i32],
) {
    let dx = (x1 - x0) / width as f32;
    let dy = (y1 - y0) / height as f32;

    let xs: Vec<_> = (0..width).map(|i| x0 + i as f32 * dx).collect();

    for j in 0..height {
        let y = y0 + j as f32 * dy;
        for (i, chunk) in xs.chunks_exact(f32x8::lanes()).enumerate() {
            let res = kernel(f32x8::from_slice_unaligned(chunk), f32x8::splat(y), count);
            let offset = (j * width) as usize + i * f32x8::lanes();
            for n in 0..f32x8::lanes() {
                output[offset + n] = res.extract(n);
            }
        }
    }
}
