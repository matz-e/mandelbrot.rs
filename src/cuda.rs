use crate::config::Domain;
use rustacuda::memory::DeviceBuffer;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

fn launch_cuda(d: Domain, output: &mut [i32], count: i32) -> Result<(), Box<dyn Error>> {
    let module_data = CString::new(include_str!("cuda.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let dx = (d.x1 - d.x0) / d.width as f32;
    let dy = (d.y1 - d.y0) / d.height as f32;

    let mut result: DeviceBuffer<i32>;

    unsafe {
        result = DeviceBuffer::zeroed(output.len())?;

        // <<<blocks, threads, shared memory, stream>>>
        launch!(module.kernel<<<((d.width / 16) as u32, (d.height / 16) as u32), (16, 16), 0, stream>>>(
            result.as_device_ptr(),
            d.x0,
            dx,
            d.y0,
            dy,
            d.width,
            count
        ))?;
    }

    stream.synchronize()?;
    result.copy_to(output)?;

    Ok(())
}

pub fn init() -> Result<Context, Box<dyn Error>> {
    rustacuda::init(CudaFlags::empty())?;

    let device = Device::get_device(0)?;
    let context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    Ok(context)
}

pub fn mandelbrot(d: Domain, count: i32, output: &mut [i32]) {
    launch_cuda(d, output, count).unwrap();
}
