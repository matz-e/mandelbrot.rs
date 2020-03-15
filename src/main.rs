mod complex;
mod serial;
mod simd;

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

fn main() {
    let width: usize = 768;
    let height: usize = 512;
    let x0: f32 = -2.0;
    let x1: f32 = 1.0;
    let y0: f32 = -1.0;
    let y1: f32 = 1.0;
    let count: i32 = 256;
    let iterations = 3;

    let mut buf = vec![0i32; width * height];

    for _ in 0..iterations {
        let start = tsc();
        serial::mandelbrot(
            x0,
            y0,
            x1,
            y1,
            width as i32,
            height as i32,
            count,
            &mut buf[..],
        );
        let end = tsc();
        println!(
            "Time: {} megacycles",
            (end - start) as f32 / (1024.0 * 1024.0)
        );
    }

    save(&buf[..], width, height, "mandelbrot.ppm");

    for _ in 0..iterations {
        let start = tsc();
        complex::mandelbrot(
            x0,
            y0,
            x1,
            y1,
            width as i32,
            height as i32,
            count,
            &mut buf[..],
        );
        let end = tsc();
        println!(
            "Time using complex: {} megacycles",
            (end - start) as f32 / (1024.0 * 1024.0)
        );
    }

    save(&buf[..], width, height, "mandelbrot_complex.ppm");

    for _ in 0..iterations {
        let start = tsc();
        simd::mandelbrot(
            x0,
            y0,
            x1,
            y1,
            width as i32,
            height as i32,
            count,
            &mut buf[..],
        );
        let end = tsc();
        println!(
            "Time using simd: {} megacycles",
            (end - start) as f32 / (1024.0 * 1024.0)
        );
    }

    save(&buf[..], width, height, "mandelbrot_simd.ppm");
}
