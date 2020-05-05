#[cfg(feature = "gpu")]
#[macro_use] extern crate rustacuda;

mod complex;
mod config;
#[cfg(feature = "gpu")]
mod cuda;
mod serial;
mod simd;

use config::Domain;

fn tsc() -> u64 {
    let n: u64;
    unsafe {
        n = core::arch::x86_64::_rdtsc();
    }
    n
}

fn save(buf: &[i32], width: usize, height: usize, filename: &str) {
    let mut imgbuf: image::RgbImage = image::ImageBuffer::new(width as u32, height as u32);
    imgbuf.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
        let index = y * width as u32 + x;
        if buf[index as usize] & 0x1 == 0x1 {
            *pixel = image::Rgb([240, 240, 240])
        } else {
            *pixel = image::Rgb([20, 20, 20])
        }
    });
    imgbuf.save(filename).unwrap();
    println!("Wrote '{}'", filename);
}

type Mandelbrot = fn(Domain, i32, &mut [i32]);

fn benchmark(
    fct: Mandelbrot,
    name: &str,
    d: Domain,
    count: i32,
    iterations: usize,
    mut output: &mut [i32],
) {
    for _ in 0..iterations {
        let start = tsc();
        fct(d, count, &mut output);
        let end = tsc();
        println!(
            "Time of {} run: {: >width$.3} megacycles",
            name,
            (end - start) as f32 / (1024.0 * 1024.0),
            width = 40 - name.len()
        );
    }

    let filename = format!("mandelbrot-{}.ppm", name);
    save(&output, d.width, d.height, &filename[..]);
}

fn main() {
    let d = Domain {
        x0: -2.0,
        x1: 1.0,
        y0: -1.0,
        y1: 1.0,
        width: 768,
        height: 512,
    };
    let count: i32 = 256;
    let iterations = 3;

    let mut buf = vec![0i32; d.width * d.height];

    benchmark(
        serial::mandelbrot,
        "serial",
        d,
        count,
        iterations,
        &mut buf[..],
    );
    benchmark(
        complex::mandelbrot,
        "complex",
        d,
        count,
        iterations,
        &mut buf[..],
    );
    benchmark(simd::mandelbrot, "simd", d, count, iterations, &mut buf[..]);
    #[cfg(feature = "gpu")]
    benchmark(cuda::mandelbrot, "cuda", d, count, iterations, &mut buf[..]);
}
