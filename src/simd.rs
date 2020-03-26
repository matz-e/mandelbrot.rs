extern crate packed_simd;

use packed_simd::*;

fn kernel(re: f32x8, im: f32x8, count: i32) -> i32x8 {
    let r2 = f32x8::splat(4.0);
    let mut counts = i32x8::splat(0);
    let mut z_re = re;
    let mut z_im = im;
    for _i in 0..count {
        let re_sqr = z_re * z_re;
        let im_sqr = z_im * z_im;
        let m = (re_sqr + im_sqr).le(r2);
        if m.none() {
            break;
        }
        counts = m.select(counts + 1, counts);
        let re_im = z_re * z_im;
        z_re = re_sqr - im_sqr + re;
        z_im = re_im + re_im + im;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_test() {
        let n = kernel(f32x8::splat(0.0), f32x8::splat(0.0), 200);
        assert_eq!(n.extract(0), 200);
        let o = kernel(f32x8::splat(2.0), f32x8::splat(0.0), 200);
        assert_eq!(o.extract(0), 1);
        let p = kernel(f32x8::splat(1.0), f32x8::splat(0.0), 200);
        assert_eq!(p.extract(0), 2);
    }
}
