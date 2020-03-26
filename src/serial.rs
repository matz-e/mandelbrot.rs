use crate::config::Domain;

fn kernel(re: f32, im: f32, count: i32) -> i32 {
    let mut new_re = re;
    let mut new_im = im;
    for i in 0..count {
        let re_sqr = new_re * new_re;
        let im_sqr = new_im * new_im;
        if re_sqr + im_sqr > 4.0 {
            return i;
        }
        let tmp = new_re * new_im;
        new_re = re_sqr - im_sqr + re;
        new_im = tmp + tmp + im;
    }
    count
}

pub fn mandelbrot(d: Domain, count: i32, output: &mut [i32]) {
    let dx = (d.x1 - d.x0) / d.width as f32;
    let dy = (d.y1 - d.y0) / d.height as f32;

    for j in 0..d.height {
        for i in 0..d.width {
            let x = d.x0 + i as f32 * dx;
            let y = d.y0 + j as f32 * dy;
            let index = (j * d.width + i) as usize;
            output[index] = kernel(x, y, count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_test() {
        let n = kernel(0.0, 0.0, 200);
        assert_eq!(n, 200);
        let o = kernel(2.0, 0.0, 200);
        assert_eq!(o, 1);
        let p = kernel(1.0, 0.0, 200);
        assert_eq!(p, 2);
    }
}
