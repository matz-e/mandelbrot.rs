extern crate num_complex;

use self::num_complex::Complex;

fn kernel(re: f32, im: f32, count: i32) -> i32 {
    let c = Complex { re, im };
    let mut z = Complex { re, im };
    for i in 0..count {
        if z.norm_sqr() > 4.0 {
            return i;
        }
        z = z * z + c;
    }
    count
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

    for j in 0..height {
        for i in 0..width {
            let x = x0 + i as f32 * dx;
            let y = y0 + j as f32 * dy;
            let index = (j * width + i) as usize;
            output[index] = kernel(x, y, count);
        }
    }
}
