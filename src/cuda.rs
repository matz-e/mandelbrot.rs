use crate::config::Domain;
use rustacuda::memory::DeviceBuffer;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use std::iter::repeat;

fn launch_cuda(
    xs: &[f32],
    ys: &[f32],
    output: &mut [i32],
    count: i32,
) -> Result<(), Box<dyn Error>> {
    rustacuda::init(CudaFlags::empty())?;

    let device = Device::get_device(0)?;
    let _context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let module_data = CString::new(include_str!("cuda.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut xs_buf = DeviceBuffer::from_slice(xs)?;
    let mut ys_buf = DeviceBuffer::from_slice(ys)?;
    let mut result: DeviceBuffer<i32>;

    unsafe {
        result = DeviceBuffer::zeroed(output.len())?;

        // <<<blocks, threads, shared memory, stream>>>
        launch!(module.kernel<<<768 * 32, 768, 0, stream>>>(
            xs_buf.as_device_ptr(),
            ys_buf.as_device_ptr(),
            result.as_device_ptr(),
            output.len(),
            count
        ))?;
    }

    stream.synchronize()?;
    result.copy_to(output)?;

    Ok(())
}

pub fn mandelbrot(d: Domain, count: i32, output: &mut [i32]) {
    let dx = (d.x1 - d.x0) / d.width as f32;
    let dy = (d.y1 - d.y0) / d.height as f32;

    let xs: Vec<_> = (0..d.height)
        .flat_map(|_j| (0..d.width).map(|i| d.x0 + i as f32 * dx))
        .collect();
    let ys: Vec<_> = (0..d.height)
        .flat_map(|j| repeat(d.y0 + j as f32 * dy).take(d.width))
        .collect();

    launch_cuda(&xs[..], &ys[..], output, count).unwrap();
}
